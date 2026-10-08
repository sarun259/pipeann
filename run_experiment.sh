#!/usr/bin/env bash
# Run a PipeANN insert+delete sliding-window experiment from a config file.
#
#   ./run_experiment.sh configs/example_tiny.conf
#
# The config is a plain KEY=value file (it is *sourced* as bash, so keep it
# to simple assignments). It maps 1:1 onto the `overall_performance` binary,
# which runs the OdinANN sliding-window update workload: at each of 100 steps
# it deletes the oldest `index_npts/STEP` tags and inserts the same number of
# fresh vectors from DATA_BIN, searching QUERY_FILE after every step and
# scoring recall against gt_<N>.bin in TRUTHSET_DIR.  See configs/README.md.
set -euo pipefail

CONFIG="${1:?usage: run_experiment.sh <config.conf>}"
source "$CONFIG"

# Binary location: override with PIPEANN_BIN, else the in-repo build.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${PIPEANN_BIN:-$REPO_ROOT/build/tests/overall_performance}"

# Fail loudly on anything missing; recall is simply skipped if GT is absent.
[[ -x "$BIN" ]]                       || { echo "binary not found: $BIN (set PIPEANN_BIN)" >&2; exit 1; }
[[ -f "$DATA_BIN" ]]                  || { echo "missing DATA_BIN: $DATA_BIN" >&2; exit 1; }
[[ -f "${INDEX_PREFIX}_disk.index" ]] || { echo "missing index: ${INDEX_PREFIX}_disk.index" >&2; exit 1; }
[[ -f "${INDEX_PREFIX}_disk.index.tags" ]] || { echo "missing tags (index is not update-capable): ${INDEX_PREFIX}_disk.index.tags" >&2; exit 1; }
[[ -f "$QUERY_FILE" ]]               || { echo "missing QUERY_FILE: $QUERY_FILE" >&2; exit 1; }
[[ -d "$TRUTHSET_DIR" ]]             || echo "WARN: TRUTHSET_DIR absent ($TRUTHSET_DIR); recall will be skipped" >&2

# Updates run against a shadow copy of the disk index; clear stale artifacts
# left by an interrupted prior run on the same prefix.
rm -f "${INDEX_PREFIX}"_shadow* "${INDEX_PREFIX}"_merge* "${INDEX_PREFIX}"temp0* 2>/dev/null || true

LOG="${LOG:-${CONFIG%.conf}.$(date +%Y%m%d_%H%M%S).log}"

# Per-event stats sink, read from the environment by overall_performance:
#   STATS_DIR            dir for raw per-event CSVs (insertions/queries/deletions
#                        + index_snapshots). Default: a .stats dir beside the log.
#   INDEX_SNAPSHOT_EVERY emit a structural IndexStats snapshot every N steps (0=off).
STATS_DIR="${STATS_DIR:-${LOG%.log}.stats}"
INDEX_SNAPSHOT_EVERY="${INDEX_SNAPSHOT_EVERY:-0}"
export STATS_DIR INDEX_SNAPSHOT_EVERY

echo "=> $BIN $TYPE $DATA_BIN $L_DISK $INDEX_PREFIX $QUERY_FILE $TRUTHSET_DIR $RECALL_AT $STEP $LSEARCH"
echo "=> logging to $LOG"
echo "=> stats CSVs -> $STATS_DIR (index snapshot every $INDEX_SNAPSHOT_EVERY steps)"
"$BIN" "$TYPE" "$DATA_BIN" "$L_DISK" "$INDEX_PREFIX" "$QUERY_FILE" "$TRUTHSET_DIR" \
       "$RECALL_AT" "$STEP" $LSEARCH 2>&1 | tee "$LOG"
