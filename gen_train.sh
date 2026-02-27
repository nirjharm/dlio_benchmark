#!/bin/bash
#
# 2026, Nj (Nirjhar) Mukherjee, CMU, PDL
#

# Default values
FILETYPE="P"
TARGET="gen"
MODEL="megatron"
NSTEP="1000"
BATCHSIZE="16"
NPROC="1"
OUTPUT_DIR=""

# Function to display help
usage() {
    echo "Usage: $0 [-filetype P|V] [-target gen|train] [-model dlrm|megatron|bert] [-nstep N] [-batchsize B] [-nproc P] [-o output_dir] [-h]"
    echo
    echo "Options:"
    echo "  -filetype   Type of dataset: P = Parquet, V = Vortex (default: P)"
    echo "  -target     Workflow target: gen or train (default: gen)"
    echo "  -model      Model type: dlrm, megatron, bert (default: megatron)"
    echo "  -nstep      Number of training steps (default: 1000)"
    echo "  -batchsize  Batch size (default: 16)"
    echo "  -nproc      Number of MPI processes (passed to mpirun -np) (default: 1)"
    echo "  -o          Output directory (default: outputs/<prefix>_<model>_run)"
    echo "  -h          Show this help message"
    echo
    echo "Examples:"
    echo "  $0 -filetype P -target gen -model megatron -nstep 2000 -batchsize 32"
    echo "  $0 -filetype V -target train -model bert -o outputs/custom_dir"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -filetype)
            FILETYPE="$2"
            shift
            shift
            ;;
        -target)
            TARGET="$2"
            shift
            shift
            ;;
        -model)
            MODEL="$2"
            shift
            shift
            ;;
        -nstep)
            NSTEP="$2"
            shift
            shift
            ;;
        -nproc)
            NPROC="$2"
            shift
            shift
            ;;
        -batchsize)
            BATCHSIZE="$2"
            shift
            shift
            ;;
        -o)
            OUTPUT_DIR="$2"
            shift
            shift
            ;;
        -h)
            usage
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

# Determine workload prefix
if [[ "$FILETYPE" == "P" ]]; then
    PREFIX="_pq"
elif [[ "$FILETYPE" == "V" ]]; then
    PREFIX="_vx"
else
    echo "Invalid filetype: $FILETYPE (use P or V)"
    exit 1
fi
# Determine model name
if [[ "$MODEL" == "dlrm" ]]; then
    MODEL_NAME="dlrm"
elif [[ "$MODEL" == "megatron" ]]; then
    MODEL_NAME="megatron_deepspeed_LLNL"
elif [[ "$MODEL" == "bert" ]]; then
    MODEL_NAME="bert_v100"
else
    echo "Invalid model: $MODEL (use dlrm, megatron, or bert)"
    exit 1
fi
# Determine workflow flags
GENERATE_DATA="False"
TRAIN="False"
if [[ "$TARGET" == "gen" ]]; then
    GENERATE_DATA="True"
elif [[ "$TARGET" == "train" ]]; then
    TRAIN="True"
else
    echo "Invalid target: $TARGET (use gen or train)"
    exit 1
fi

# Build output directory (use -o if provided)
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="outputs/${PREFIX}_${MODEL_NAME}_run"
fi

# Run the command
echo "Running DLIO with workload=${PREFIX}_${MODEL_NAME} (np=${NPROC})"
echo "Output directory: $OUTPUT_DIR"

VENV_PY="/users/nmukherj/venvs/duckdb/bin/python"

export DFTRACER_ENABLE=1

mpirun -np ${NPROC} $VENV_PY -m dlio_benchmark.main \
    workload=${PREFIX}_${MODEL_NAME} \
    ++workload.workflow.generate_data=${GENERATE_DATA} \
    ++workload.workflow.train=${TRAIN} \
    ++workload.output_dir=${OUTPUT_DIR} \
    ++workload.train.total_training_steps=${NSTEP} \
    ++workload.reader.batch_size=${BATCHSIZE} \
    ++hydra.run.dir=${OUTPUT_DIR}

