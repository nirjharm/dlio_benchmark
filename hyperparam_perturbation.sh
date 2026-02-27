#!/bin/bash
sleep 5
# Script to run grid sweeps of gen_train.sh and save outputs
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN_SCRIPT="${SCRIPT_DIR}/gen_train.sh"
BASE_DIR="${SCRIPT_DIR}/hyperparameter_tuning_results"

# Disk to monitor (device name under /sys/block)
DISK="nvme3n1"

# If set to True, only run when the output folder is missing.
# Set by exporting ERROR_ONLY=True in the environment or editing this file.
ERROR_ONLY="${ERROR_ONLY:-False}"

mkdir -p "$BASE_DIR"

idx=0
for filetype in V P; do
    for batch in 16 32 64; do
        for nproc in 1; do
            # compute Full-1: floor(100000 / (batch * nproc)) - 1
            full_minus1=$((100000 / (batch * nproc) - 1))
            if [ $full_minus1 -lt 1 ]; then
                full_minus1=1
            fi
            for kind in full half quarter; do
                case "$kind" in
                    full)
                        nstep=$full_minus1
                        ;;
                    half)
                        nstep=$(( full_minus1 / 2 ))
                        ;;
                    quarter)
                        nstep=$(( full_minus1 / 4 ))
                        ;;
                    *)
                        nstep=$full_minus1
                        ;;
                esac
                if [ $nstep -lt 1 ]; then
                    nstep=1
                fi
                idx=$((idx+1))
                uniq="run_f${filetype}_b${batch}_s${nstep}_p${nproc}_i${idx}"
                outdir="${BASE_DIR}/${uniq}"
                # If ERROR_ONLY is True, skip this run when output dir already exists
                if [ "${ERROR_ONLY}" = "True" ] || [ "${ERROR_ONLY}" = "true" ]; then
                    if [ -d "${outdir}" ]; then
                        echo "ERROR_ONLY is set and ${outdir} exists — skipping run."
                        continue
                    fi
                fi
                mkdir -p "$outdir"

                #ensure output dir is empty
                # Removes all files and directories inside the directory specified by the 'outdir' variable.
                # The ':?' syntax in "${outdir:?}" is a safety feature that causes the script to exit with an error
                # if 'outdir' is unset or null, preventing accidental deletion of unintended directories.
                rm -rf "${outdir:?}"/*

                echo "Dropping pagecache/etc"
                sudo bash -c 'echo 3 > /proc/sys/vm/drop_caches'
                sleep 1

                echo "Running: filetype=${filetype}, batch=${batch}, nstep=${nstep} -> ${outdir}"
                # Start background disk monitor: write elapsed_milliseconds,delta_sectors to nj_disktrace
                (
                    # get starting sectors (field 3)
                    read _ _ start_sectors _ < /sys/block/${DISK}/stat 2>/dev/null || start_sectors=0

                    start_ns=$(date +%s%N)

                    printf "0,0,0\n" > "${outdir}/nj_disktrace"

                    while true; do
                        # ---- Disk (no awk) ----
                        read _ _ cur_sectors _ < /sys/block/${DISK}/stat 2>/dev/null || cur_sectors=0

                        cur_ns=$(date +%s%N)
                        elapsed_ms=$(( (cur_ns - start_ns) / 1000000 ))
                        delta=$((cur_sectors - start_sectors))

                        # ---- DRAM (no awk) ----
                        while read key value _; do
                            case "$key" in
                                MemTotal:) t=$value ;;
                                MemAvailable:) a=$value ;;
                            esac
                        done < /proc/meminfo

                        dram_used_mb=$(( (t - a) / 1024 ))

                        printf "%d,%d,%d\n" "$elapsed_ms" "$delta" "$dram_used_mb" >> "${outdir}/nj_disktrace"

                        sleep 0.5
                    done
                ) &
                monitor_pid=$!
                # Run gen_train.sh and capture output
                "$GEN_SCRIPT" -filetype "$filetype" -target train -model dlrm -nstep "$nstep" -batchsize "$batch" -nproc "$nproc" -o "$outdir" 2>&1 | tee "${outdir}/run.log"
                # stop monitor
                if [ -n "${monitor_pid:-}" ]; then
                    kill ${monitor_pid} 2>/dev/null || true
                    wait ${monitor_pid} 2>/dev/null || true
                fi
            done
        done
	done
done

echo "All runs complete. Results in ${BASE_DIR}"
