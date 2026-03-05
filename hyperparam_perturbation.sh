#!/bin/bash
sleep 5
# Script to run grid sweeps of gen_train.sh and save outputs
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN_SCRIPT="${SCRIPT_DIR}/gen_train.sh"
BASE_DIR="${SCRIPT_DIR}/hyperparameter_tuning_results"

# If set to True, only run when the output folder is missing.
ERROR_ONLY="${ERROR_ONLY:-False}"

mkdir -p "$BASE_DIR"

# Dataset parameters
NUM_SAMPLES_PER_FILE=100000    # <-- adjust if dataset changes
NUM_FILES_TRAIN=1
TOTAL_SAMPLES=$(( NUM_SAMPLES_PER_FILE * NUM_FILES_TRAIN ))
RECORD_SIZE=34684              # in bytes, adjust for your dataset
IO_DISK="/dev/nvme3n1"          # disk device to monitor for I/O stats
IO_DISK_NAME="$(basename "$IO_DISK")"

# Default DLIO parameters for scaling compute_time
DEFAULT_COMPUTE=0.064296
DEFAULT_BATCH=1
DEFAULT_RECORD_SIZE=671088640

idx=0
for filetype in P V; do
    for batch in 16; do
        for nproc in 8; do

            # calculate max steps per process
            max_steps=$(( TOTAL_SAMPLES / (batch * nproc) ))
            if [ $max_steps -lt 1 ]; then
                max_steps=1
            fi

            # step variations: full, half, quarter
            declare -A step_map
            step_map[full]=$(( max_steps - 1 ))
            step_map[half]=$(( max_steps / 2 ))
            step_map[quarter]=$(( max_steps / 4 ))

            # ensure min 1 step
            for key in "${!step_map[@]}"; do
                if [ ${step_map[$key]} -lt 1 ]; then
                    step_map[$key]=1
                fi
            done

            # loop over step types
            for kind in full; do # half quarter; do
                nstep=${step_map[$kind]}
                idx=$((idx+1))
                uniq="run_f${filetype}_b${batch}_s${nstep}_p${nproc}_i${idx}"
                outdir="${BASE_DIR}/${uniq}"

                # skip existing output if ERROR_ONLY
                if [[ "${ERROR_ONLY}" =~ ^[Tt]rue$ ]] && [ -d "${outdir}" ]; then
                    echo "ERROR_ONLY is set and ${outdir} exists — skipping run."
                    continue
                fi

                mkdir -p "$outdir"
                rm -rf "${outdir:?}"/*

                echo "Dropping pagecache/etc"
                sudo bash -c 'echo 3 > /proc/sys/vm/drop_caches'
                sleep 1

                # compute synthetic computation_time for this batch
                compute_time=$(echo "$DEFAULT_COMPUTE * ($batch / $DEFAULT_BATCH) * ($RECORD_SIZE / $DEFAULT_RECORD_SIZE)" | bc -l)

                echo "Running: filetype=${filetype}, batch=${batch}, nstep=${nstep}, nproc=${nproc}, ct=${compute_time} -> ${outdir}"

                # start background I/O + DRAM monitor on disk device before training
                start_ns=$(date +%s%N)
                (
                    printf "0,0,0\n" > "${outdir}/nj_disktrace"   # timestamp_ms, read_bytes, dram_used_mb

                    # baseline read bytes at monitor start so trace is delta-from-zero
                    if [ -r "/sys/block/${IO_DISK_NAME}/stat" ]; then
                        initial_read_bytes=$(awk '{print $3 * 512}' /sys/block/${IO_DISK_NAME}/stat)
                    else
                        initial_read_bytes=0
                    fi

                    while true; do
                        cur_ns=$(date +%s%N)
                        elapsed_ms=$(( (cur_ns - start_ns) / 1000000 ))

                        # read I/O stats from disk device (sectors read * 512 = bytes)
                        if [ -r "/sys/block/${IO_DISK_NAME}/stat" ]; then
                            current_read_bytes=$(awk '{print $3 * 512}' /sys/block/${IO_DISK_NAME}/stat)
                            read_bytes=$(( current_read_bytes - initial_read_bytes ))
                            if [ "$read_bytes" -lt 0 ]; then
                                read_bytes=0
                            fi
                        else
                            read_bytes=0
                        fi

                        # DRAM usage
                        while read key value _; do
                            case "$key" in
                                MemTotal:) t=$value ;;
                                MemAvailable:) a=$value ;;
                            esac
                        done < /proc/meminfo
                        dram_used_mb=$(( (t - a) / 1024 ))

                        printf "%d,%d,%d\n" "$elapsed_ms" "$read_bytes" "$dram_used_mb" >> "${outdir}/nj_disktrace"
                        sleep 0.5
                    done
                ) &
                monitor_pid=$!

                # start gen_train.sh in background
                "$GEN_SCRIPT" -filetype "$filetype" -target train \
                    -model dlrm -nstep "$nstep" -batchsize "$batch" -nproc "$nproc" \
                    -ct "$compute_time" \
                    -o "$outdir" 2>&1 | tee "${outdir}/run.log" &
                train_pid=$!

                # wait for train to finish
                wait $train_pid

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