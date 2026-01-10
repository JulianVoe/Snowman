#!/bin/bash
set -euo pipefail

# --- configuration ---------------------------------------------------------
N=${N:-1024}
MEN=${MEN:-4}
# space separated list of "procs threads"
CFGS=${CFGS:-"4 3 12 1 1 12"}

OUT="intel_profile_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
echo "Writing results to: $OUT"

# --- environment / build ---------------------------------------------------
if command -v module >/dev/null 2>&1; then
  module purge
  module load intel || true
  module load intel-compilers || true
  module load impi || true
fi

make clean
CXX=mpiicpc CXXFLAGS="-O3 -std=c++17 -qopenmp" make

mv snowman "$OUT"
cd "$OUT"

# --- Intel MPI process pinning (hybrid-aware) ------------------------------
export I_MPI_PIN=1
export I_MPI_PIN_CELL=core
export I_MPI_PIN_DOMAIN=omp:compact
export I_MPI_PIN_ORDER=compact
export I_MPI_DEBUG=5
export I_MPI_PIN_RESPECT_CPUSET=1

# --- OpenMP thread pinning inside each MPI rank ----------------------------
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_DISPLAY_ENV=VERBOSE

secs() {
    local nprocs=$1
    local nthreads=$2

    export OMP_NUM_THREADS="$nthreads"

    local tag="${nprocs}procs_${nthreads}threads"
    local log="binding_${tag}.log"

    echo "Running config: procs=${nprocs}, threads=${nthreads} (log: ${log})"

    ( time mpirun -np "$nprocs" ./snowman "$N" "$MEN" ) >"$log" 2>&1

    awk '/^real/ { sub(/m/, " "); sub(/s/, ""); print $2*60 + $3 }' "$log"
}

CSV="runs_intel.csv"
echo "threads,procs,seconds" > "$CSV"

set -- $CFGS
while [ $# -gt 0 ]; do
  procs=$1
  threads=$2
  t=$(secs "$procs" "$threads")
  echo "$threads,$procs,$t" | tee -a "$CSV"
  shift 2
done

echo "All runs finished. CSV file: $CSV"
echo "Binding / mapping logs:"
ls binding_*.log
