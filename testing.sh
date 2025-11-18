#!/usr/bin/env bash
set -euo pipefail

# ---------- configuration (edit once, not every run) ----------

# instrumented binary from "make instr"
APP_DEFAULT=./snowman-instr 1024 4

# default MPI ranks
NP_DEFAULT=8

# memory hint for tracing; you can adjust after inspecting scorep-score output
SCOREP_TOTAL_MEMORY_DEFAULT=3G

# MPI groups we care about
MPI_GROUPS_DEFAULT="cg,coll,p2p,xnonblock"

# experiment directory names
PROFILE_DIR_DEFAULT=scorep_profile
TRACE_DIR_DEFAULT=scorep_trace

# ---------- argument parsing (optional overrides) ----------

APP="$APP_DEFAULT"
NP="$NP_DEFAULT"
SCOREP_TOTAL_MEMORY="$SCOREP_TOTAL_MEMORY_DEFAULT"
PROFILE_DIR="$PROFILE_DIR_DEFAULT"
TRACE_DIR="$TRACE_DIR_DEFAULT"
MPI_GROUPS="$MPI_GROUPS_DEFAULT"

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  -e PATH   Instrumented executable (default: $APP_DEFAULT)
  -n N      Number of MPI ranks     (default: $NP_DEFAULT)
  -m MEM    SCOREP_TOTAL_MEMORY     (default: $SCOREP_TOTAL_MEMORY_DEFAULT)
  -p DIR    Profiling experiment dir (default: $PROFILE_DIR_DEFAULT)
  -t DIR    Tracing experiment dir   (default: $TRACE_DIR_DEFAULT)
  -g GROUPS SCOREP_MPI_ENABLE_GROUPS (default: $MPI_GROUPS_DEFAULT)
  -h        Show this help
EOF
}

while getopts "e:n:m:p:t:g:h" opt; do
    case "$opt" in
        e) APP="$OPTARG" ;;
        n) NP="$OPTARG" ;;
        m) SCOREP_TOTAL_MEMORY="$OPTARG" ;;
        p) PROFILE_DIR="$OPTARG" ;;
        t) TRACE_DIR="$OPTARG" ;;
        g) MPI_GROUPS="$OPTARG" ;;
        h) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
done

if [[ ! -x "$APP" ]]; then
    echo "Error: executable '$APP' not found or not executable." >&2
    exit 1
fi

# -----------------------------------------
make instr

# ---------- step 1: profiling run ----------

echo "==> Profiling run (Score-P profiling only) ..."
SCOREP_EXPERIMENT_DIRECTORY="$PROFILE_DIR" \
SCOREP_ENABLE_PROFILING=true \
SCOREP_ENABLE_TRACING=false \
SCOREP_MPI_ENABLE_GROUPS="$MPI_GROUPS" \
mpirun -n "$NP" "$APP"

PROFILE_FILE="$PROFILE_DIR/profile.cubex"
if [[ ! -f "$PROFILE_FILE" ]]; then
    echo "Error: $PROFILE_FILE not found. Did the run fail?" >&2
    exit 1
fi

echo "==> Running scorep-score to derive filter and memory hints ..."
FILTER_FILE="scorep.filter"
scorep-score -f "$FILTER_FILE" "$PROFILE_FILE"

echo
echo "scorep-score finished."
echo "  -> Filter file: $FILTER_FILE"
echo "  -> Inspect the above output for a recommended SCOREP_TOTAL_MEMORY"
echo "     (currently using $SCOREP_TOTAL_MEMORY_DEFAULT, override via -m if needed)."
echo

# ---------- step 2: tracing run ----------

echo "==> Tracing run (Score-P tracing only) ..."
SCOREP_EXPERIMENT_DIRECTORY="$TRACE_DIR" \
SCOREP_ENABLE_PROFILING=false \
SCOREP_ENABLE_TRACING=true \
SCOREP_FILTERING_FILE="$FILTER_FILE" \
SCOREP_MPI_ENABLE_GROUPS="$MPI_GROUPS" \
SCOREP_TOTAL_MEMORY="$SCOREP_TOTAL_MEMORY" \
mpirun -np "$NP" "$APP"

TRACE_FILE="$TRACE_DIR/traces.otf2"
if [[ -f "$TRACE_FILE" ]]; then
    echo
    echo "Trace generated at: $TRACE_FILE"
    echo "Open it with Vampir, e.g.:"
    echo "  vampir $TRACE_FILE"
else
    echo "Warning: $TRACE_FILE not found. Check $TRACE_DIR for details." >&2
fi

