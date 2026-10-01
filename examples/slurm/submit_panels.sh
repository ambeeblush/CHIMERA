#!/bin/bash
#SBATCH --job-name=chimera_panels
#SBATCH --output=logs/chimera_%A_%a.out
#SBATCH --error=logs/chimera_%A_%a.err
#SBATCH --array=1-4                  # numero di task: vedi "Quanti task" sotto
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
# #SBATCH --partition=gpuq            # scommenta se hai una partizione GPU
# #SBATCH --gres=gpu:1                # clesperanto usa OpenCL: senza GPU
#                                     # l'interpolazione lineare ripiega su CPU

set -euo pipefail

# --------------------------------------------------------------------------- #
# Configurazione: gli unici valori da toccare
# --------------------------------------------------------------------------- #

PROJECT=CHIMERA
ENV_PATH=chimera-env

CONFIG="${PROJECT}/configs/default.yaml"
MANIFEST="${PROJECT}/input_csv/input_csv.csv"
OUTPUT_DIR="${PROJECT}/out/$(date +%Y-%m-%d)_${SLURM_ARRAY_JOB_ID:-manual}"
DATA_ROOT=analysis_output

# --------------------------------------------------------------------------- #

mkdir -p logs "${OUTPUT_DIR}"

# L'ambiente: adatta alla tua installazione (conda, venv o module)
source "${ENV_PATH}/bin/activate" 2>/dev/null || {
    module load python/3.11
    source "${ENV_PATH}/bin/activate"
}

export PYTHONPATH="${PROJECT}/src:${PYTHONPATH:-}"

# Con un job array, --shard e --num-shards si leggono da soli da
# SLURM_ARRAY_TASK_ID e SLURM_ARRAY_TASK_COUNT: non vanno passati.
echo "=== task ${SLURM_ARRAY_TASK_ID:-0} di ${SLURM_ARRAY_TASK_COUNT:-1} su $(hostname) ==="
echo "=== output: ${OUTPUT_DIR} ==="

python -m chimera.cli \
    --config        "${CONFIG}" \
    --manifest      "${MANIFEST}" \
    --output-dir    "${OUTPUT_DIR}" \
    --data-root     "${DATA_ROOT}" \
    --n-frames      36 \
    --label-size    30 \
    --scalebar-um   10 \
    --fps           6 \
    --quality       8 \
    --skip-existing \
    --on-error      continue \
    --log-level     INFO

status=$?

# Exit code: 0 tutto bene, 1 qualche riga fallita, 2 errore di configurazione.
# Senza propagarlo, SLURM segnerebbe COMPLETED anche una run senza output.
echo "=== task terminato con codice ${status} ==="
exit ${status}
