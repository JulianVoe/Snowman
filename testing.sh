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
module load LinaroForge/25.0.3-GCCcore-13.2.0-linux-x86_64
module load Score-P

# ---------- configuration (edit once, not every run) ----------

# instrumented binary from "make instr"
APP_DEFAULT=./snowman-instr

# default MPI ranks
NP_DEFAULT=8

# default app args: "snowman-instr 1024 4"
WIDTH_DEFAULT=1024
OTHER_DEFAULT=4

# default memory hint for tracing; adjust after first scorep-score run
SCOREP_TOTAL_MEMORY_DEFAULT=512M

# MPI groups we care about
MPI_GROUPS_DEFAULT="cg,coll,p2p"

# experiment directory names
PROFILE_DIR_DEFAULT=scorep_profile
TRACE_DIR_DEFAULT=scorep_trace

# filter file name (fixed; created by scorep-score -g)
FILTER_FILE=initial_scorep.filter


# ---------- argument parsing (optional overrides) ----------

APP="$APP_DEFAULT"
NP="$NP_DEFAULT"
WIDTH="$WIDTH_DEFAULT"
OTHER="$OTHER_DEFAULT"
SCOREP_TOTAL_MEMORY="$SCOREP_TOTAL_MEMORY_DEFAULT"
PROFILE_DIR="$PROFILE_DIR_DEFAULT"
TRACE_DIR="$TRACE_DIR_DEFAULT"
MPI_GROUPS="$MPI_GROUPS_DEFAULT"

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  -e PATH   Instrumented executable          (default: $APP_DEFAULT)
  -n N      Number of MPI ranks              (default: $NP_DEFAULT)
  -w N      First app argument (width)       (default: $WIDTH_DEFAULT)
  -o N      Second app argument (other)      (default: $OTHER_DEFAULT)
  -m MEM    SCOREP_TOTAL_MEMORY              (default: $SCOREP_TOTAL_MEMORY_DEFAULT)
  -p DIR    Profiling experiment dir         (default: $PROFILE_DIR_DEFAULT)
  -t DIR    Tracing experiment dir           (default: $TRACE_DIR_DEFAULT)
  -g GROUPS SCOREP_MPI_ENABLE_GROUPS         (default: $MPI_GROUPS_DEFAULT)
  -h        Show this help
EOF
}

while getopts "e:n:w:o:m:p:t:g:h" opt; do
    case "$opt" in
        e) APP="$OPTARG" ;;
        n) NP="$OPTARG" ;;
        w) WIDTH="$OPTARG" ;;
        o) OTHER="$OPTARG" ;;
        m) SCOREP_TOTAL_MEMORY="$OPTARG" ;;
        p) PROFILE_DIR="$OPTARG" ;;
        t) TRACE_DIR="$OPTARG" ;;
        g) MPI_GROUPS="$OPTARG" ;;
        h) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
done


# ---------- build: make instr ----------

echo "==> Running 'make instr' ..."
make instr

if [[ ! -x "$APP" ]]; then
    echo "Error: executable '$APP' not found or not executable after 'make instr'." >&2
    exit 1
fi


# ---------- optional step: profiling run to generate filter ----------

if [[ ! -f "$FILTER_FILE" ]]; then
    echo "==> No filter file '$FILTER_FILE' found."
    echo "==> Running profiling step to generate an initial filter..."

    SCOREP_EXPERIMENT_DIRECTORY="$PROFILE_DIR" \
    SCOREP_ENABLE_PROFILING=true \
    SCOREP_ENABLE_TRACING=false \
    SCOREP_MPI_ENABLE_GROUPS="$MPI_GROUPS" \
    mpirun -np "$NP" "$APP" "$WIDTH" "$OTHER"

    PROFILE_FILE="$PROFILE_DIR/profile.cubex"
    if [[ ! -f "$PROFILE_FILE" ]]; then
        echo "Error: $PROFILE_FILE not found after profiling run. Did the run fail?" >&2
        exit 1
    fi

    echo "==> Running scorep-score -g to create '$FILTER_FILE' ..."
    # Heuristics: filter small, frequently-called USR regions
    scorep-score -g bufferpercent=1,timepervisit=1,type=usr "$PROFILE_FILE"

    if [[ ! -f "$FILTER_FILE" ]]; then
        echo "Error: '$FILTER_FILE' was not created by scorep-score -g." >&2
        exit 1
    fi

    echo
    echo "Filter created: $FILTER_FILE"
    echo "You can edit it by hand if you want finer control."
    echo "Also check the scorep-score output above for a better SCOREP_TOTAL_MEMORY hint."
    echo
else
    echo "==> Using existing filter file '$FILTER_FILE' – skipping profiling step."
fi


# ---------- tracing run ----------

echo "==> Tracing run (Score-P tracing only) ..."
SCOREP_EXPERIMENT_DIRECTORY="$TRACE_DIR" \
SCOREP_ENABLE_PROFILING=false \
SCOREP_ENABLE_TRACING=true \
SCOREP_FILTERING_FILE="$FILTER_FILE" \
SCOREP_MPI_ENABLE_GROUPS="$MPI_GROUPS" \
SCOREP_TOTAL_MEMORY="$SCOREP_TOTAL_MEMORY" \
mpirun -np "$NP" "$APP" "$WIDTH" "$OTHER"

TRACE_FILE="$TRACE_DIR/traces.otf2"
if [[ -f "$TRACE_FILE" ]]; then
    echo
    echo "Trace generated at: $TRACE_FILE"
    echo "Open it with Vampir, e.g.:"
    echo "  vampir $TRACE_FILE"
else
    echo "Warning: $TRACE_FILE not found. Check contents of '$TRACE_DIR' for details." >&2
fi
