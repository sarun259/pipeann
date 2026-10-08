#!/usr/bin/env bash
# Static (no mem index, no inserts) disk search on a prebuilt PipeANN index,
# reporting the QueryStats aggregates (AvgLat / P99 / Mean Hops / Mean IOs) plus
# Recall@10 against an exact ground truth we compute once.
#
# Usage:  tests/run_static_search.sh [sift1m|sift100m]   (default sift1m)
#
# Indices are uint8 / dim-128 / L2 / PQ, owned by another user and read-only to
# us. We read them IN PLACE -- the reader now honors its enable_writes flag and
# opens O_RDONLY for search, so no multi-GB copy is needed (essential at 100M).
set -euo pipefail

DATASET="${1:-sift1m}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
BUILD="$REPO/build"
SCRATCH=/tmp/pipeann_static
DATASETS_DIR=/mnt/hdd2t/tyang425/fnct-db-eval/datasets

case "$DATASET" in
  sift1m)
    INDEX_PREFIX=/mnt/ssd4t/tyang425/pipeann/idx
    BASE_BIN="$DATASETS_DIR/sift.1M.128.parlay"       # 1M uint8, DiskANN .bin
    ;;
  sift100m)
    # WARNING: pipeann002_sift is the only 100M PipeANN index available, but it is
    # a DIRTIED DYNAMIC SNAPSHOT, not a clean static index. It was loaded as a 100M
    # base and then driven through inserts (id2loc grew to ~2.25e8); PipeANN updates
    # the disk graph in place, so neighbor lists now reference ids with no PQ entry
    # and no saved id->loc mapping -> static search segfaults. This branch is
    # scaffolding for when a CLEAN static 100M index exists. Override to attempt:
    #   ALLOW_DIRTY_100M=1 tests/run_static_search.sh sift100m
    INDEX_PREFIX=/mnt/ssd4t/tyang425/fnct-db-eval/index/pipeann002_sift
    BASE_BIN="$DATASETS_DIR/sift.100M.128.u8bin"      # 100M uint8, DiskANN .bin
    if [[ "${ALLOW_DIRTY_100M:-0}" != 1 ]]; then
      echo "refusing sift100m: $INDEX_PREFIX is a dirtied dynamic snapshot (see comment)."
      echo "need a clean static 100M index. set ALLOW_DIRTY_100M=1 to try anyway."
      exit 1
    fi
    ;;
  *) echo "unknown dataset '$DATASET' (use sift1m|sift100m)"; exit 1 ;;
esac

RAW_QUERY=/mnt/hdd2t/ann-data-insert/sift.10K.128.u8.query
QUERY_BIN="$SCRATCH/sift.10K.128.u8.bin"
GT_FILE="$SCRATCH/gt_${DATASET}_k100.bin"
GT_K=100
DIM=128

mkdir -p "$SCRATCH"

# --- build (see standalone-build-gotchas memory for cblas/aio workarounds) ----
SHIM=/tmp/cblas_shim
MKL_RT=$(ldconfig -p | awk '/libmkl_rt.so/{print $NF; exit}')
if [[ -n "${MKL_RT:-}" && ! -e "$SHIM/libcblas.so" ]]; then
  mkdir -p "$SHIM" && ln -sf "$MKL_RT" "$SHIM/libcblas.so"
fi
export LIBRARY_PATH="$SHIM:${LIBRARY_PATH:-}"

# Configure every run (cheap, idempotent) as its own step so newly added targets
# land in the regenerated Makefile before make resolves them.
cmake -S "$REPO" -B "$BUILD" -DIO_ENGINE=aio >/dev/null
make -C "$BUILD" search_disk_index compute_groundtruth index_metrics -j"$(nproc)"

# --- query file: prepend a DiskANN [int32 npts][int32 dim] header ------------
if [[ ! -f "$QUERY_BIN" ]]; then
  python3 - "$RAW_QUERY" "$QUERY_BIN" "$DIM" <<'PY'
import struct, sys
src, dst, dim = sys.argv[1], sys.argv[2], int(sys.argv[3])
raw = open(src, 'rb').read()
assert len(raw) % dim == 0, (len(raw), dim)
n = len(raw) // dim                       # 1 byte/elem (uint8)
open(dst, 'wb').write(struct.pack('<ii', n, dim) + raw)
print(f"wrote {dst}: npts={n} dim={dim}")
PY
fi

# --- exact ground truth, computed once (brute force over the base) -----------
# GT ids are base positions; the index has identity tags, so recall compares
# them directly against the tags the search returns. Assumes the index was built
# from BASE_BIN in the same order -- a wrong base shows up as ~0 recall.
# At 100M this is a long pass over a 12.8GB base; it prints per-batch progress.
if [[ ! -f "$GT_FILE" ]]; then
  echo "computing ground truth ($DATASET, K=$GT_K) -> $GT_FILE ..."
  "$BUILD/tests/utils/compute_groundtruth" \
    uint8 l2 "$BASE_BIN" "$QUERY_BIN" "$GT_K" "$GT_FILE" null
fi

# --- run static beam search --------------------------------------------------
#  args: type prefix nthreads beamwidth query truthset K metric nbr_type
#        search_mode mem_L  L1 L2 ...
#  search_mode 0 = beam search ; mem_L 0 = no in-memory index (pure disk/static)
echo "### $DATASET search (reading index in place at $INDEX_PREFIX) ###"
PAGES_CSV="$SCRATCH/pages_per_search_${DATASET}.csv"
VISITED_CSV="$SCRATCH/visited_per_hop_${DATASET}.csv"
rm -f "$PAGES_CSV" "$VISITED_CSV"
PAGES_PER_SEARCH_OUT="$PAGES_CSV" VISITED_PER_HOP_OUT="$VISITED_CSV" \
  "$BUILD/tests/search_disk_index" \
  uint8 "$INDEX_PREFIX" 16 4 \
  "$QUERY_BIN" "$GT_FILE" \
  10 l2 pq \
  0 0 \
  10 20 40 60 100

# --- static graph-structure metrics + plots ----------------------------------
# One snapshot now (static index); the insert/delete workload will append more.
METRICS_JSON="$SCRATCH/metrics_${DATASET}.json"
echo "### $DATASET static graph metrics ###"
"$BUILD/tests/index_metrics" \
  uint8 "$INDEX_PREFIX" l2 \
  "${SAMPLE_SIZE:-10000}" 10 "$METRICS_JSON"

# matplotlib lives in a cached venv (the system python has no pip/numpy).
VENV="$SCRATCH/venv"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "creating plot venv + installing matplotlib (one time) ..."
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --disable-pip-version-check matplotlib
fi
PLOTS="$REPO/plots/${DATASET}"
mkdir -p "$PLOTS"
"$VENV/bin/python" "$HERE/plot_metrics.py" "$METRICS_JSON" "$PAGES_CSV" "$PLOTS" "$VISITED_CSV"
