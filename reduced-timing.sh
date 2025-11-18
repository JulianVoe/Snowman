#!/bin/bash
#SBATCH --account=tmp_hpca_workshop
#SBATCH --partition=intelsr_devel
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --job-name=ex01_scaling
#SBATCH --output=ex01_%j.out
#SBATCH --error=ex01_%j.err
#SBATCH --exclusive

set -euo pipefail
set -x

unset SLURM_EXPORT_ENV
module purge
module load OpenMPI/4.1.6-GCC-13.2.0
module load LinaroForge/25.0.3-GCCcore-13.2.0-linux-x86_64

make clean && make

OUT="outputs_${SLURM_JOB_ID}"; REPD="$OUT/reports"
mkdir -p "$OUT" "$REPD"

mv snowman "$OUT"

# ----------------------- params -----------------------
TASKS=(8 16 24 32 48 64 96)

SS_N=1024   # strong-scaling grid
SS_MEN=4

WS_N=128    # weak-scaling per-rank baseline (linear size = sqrt in pixels)
WS_MEN=3
# ------------------------------------------------------

SS_CSV="$OUT/strong_scaling.csv"
WS_CSV="$OUT/weak_scaling.csv"
echo "workers,grid,#snowmen,seconds" > "$SS_CSV"
echo "workers,grid,#snowmen,seconds" > "$WS_CSV"

secs() { #Times runtime and then converts "real" to seconds
    ( time mpirun -n "$1" "$OUT/snowman" "$2" "$3" >/dev/null ) 2>&1 \
    | awk '/^real/ { sub(/m/, " "); sub(/s/, ""); print $2*60 + $3 }'
}
pixels_per_s() { awk -v N="$1" -v t="$2" 'BEGIN{printf "%.6f", (N*N)/t}'; }
div() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.6f", a/b}'; }
mul() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.6f", a*b}'; }
round(){ awk -v x="$1" 'BEGIN{printf "%d", (x<0)?int(x-0.5):int(x+0.5)}'; }

# --- strong scaling (fixed SS_N) ---
for p in "${TASKS[@]}"; do
  t=$(secs "$p" "$SS_N" "$SS_MEN")
  echo "$p,$SS_N,$SS_MEN,$t" >> "$SS_CSV"
done

# --- weak scaling (N ~ WS_N * sqrt(p)) ---
for p in "${TASKS[@]}"; do
  N=$(round "$(div "$(awk -v n=$WS_N -v r=$p 'BEGIN{printf "%.6f", n*sqrt(r)}')" 1)")
  t=$(secs "$p" "$N" "$WS_MEN")
  echo "$p,$N,$WS_MEN,$t" >> "$WS_CSV"
done
