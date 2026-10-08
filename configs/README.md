# Config-driven insert+delete experiments

`run_experiment.sh <config>` runs PipeANN's **insert+delete sliding-window**
benchmark (`tests/overall_performance`) from a small config file. One config
describes one experiment: which index, queries, dataset, and ground truth to use.

```bash
# verified, runs in seconds:
PIPEANN_BIN=/mnt/ssd4t/sarun/pipeann-build/tests/overall_performance \
LD_LIBRARY_PATH=/mnt/ssd4t/sarun/cblas_shim \
./run_experiment.sh configs/example_tiny.conf
```

## The workload (what "insertions / deletions" mean here)

`overall_performance` does **not** take explicit insert/delete files. It runs the
OdinANN sliding window: the index starts with `index_npts` vectors, and for each
of 100 steps it

- **deletes** the oldest `index_npts/STEP` tags, and
- **inserts** the next `index_npts/STEP` vectors read from `DATA_BIN`,

then searches `QUERY_FILE` and reports QPS + recall@`RECALL_AT` against the
checkpoint ground truth. So "insertions" are the tail of `DATA_BIN` past the
initial index, and "deletions" are the oldest tags — both derived from the
window, not separate files. `DATA_BIN` must therefore hold more vectors than the
index (e.g. a 200M dataset for a 100M index doing a full turnover).

## Config fields

Plain `KEY=value` (sourced as bash — keep to simple assignments):

| Key | Meaning |
|-----|---------|
| `TYPE` | `int8` / `uint8` / `float` — must match the data. |
| `DATA_BIN` | Dataset in `.bin` format. New inserts come from here. |
| `L_DISK` | Candidate-pool `L` used while inserting. |
| `INDEX_PREFIX` | Loads `{PREFIX}_disk.index` + `{PREFIX}_disk.index.tags`. |
| `QUERY_FILE` | Queries, same dtype/dim as the data. |
| `TRUTHSET_DIR` | Dir of `gt_<N>.bin` (one per checkpoint). Missing ⇒ recall skipped. |
| `RECALL_AT` | `k` in recall@k. |
| `STEP` | Window divisor: `index_npts/STEP` vectors deleted+inserted per step. |
| `LSEARCH` | Space-separated search `L` values to sweep. |
| `STATS_DIR` | *(optional)* Dir for raw per-event stats CSVs. Unset ⇒ a `.stats` dir beside the log. |
| `INDEX_SNAPSHOT_EVERY` | *(optional)* Dump a structural `IndexStats` snapshot every N of the 100 steps (`0` = off). |

Threads are compile-time (`NUM_*_THREADS` in `overall_performance.cpp`), not config.

## Stats output (CSV)

When `STATS_DIR` is set, the run writes raw per-event CSVs there (one row per
event, so downstream analysis keeps full distributions):

| File | One row per | Columns |
|------|-------------|---------|
| `insertions.csv` | insert | `step,tag,latency_us` + `InsertionStats` fields (`n_rmw_reads,n_pages_touched,n_out_edges,n_in_edges,n_evictions,search_total_us`) |
| `queries.csv` | measured query | `step,L,query` + `QueryStats` fields (`total_us,n_ios,n_cmps,n_exact,n_hops`) |
| `deletions.csv` | lazy-delete | `step,tag,latency_us` (lazy_delete only marks; compaction stats live in `merge_deletes`) |
| `index_snapshots.csv` | metric value | long/tidy `label,n_live,n_pages,metric,bin,value`; `metric` ∈ {`outdeg`,`indeg`,`page_fill`} |

`step` is the 0-based window step (`-1` = the pre-insert baseline search). The
column names come straight from each stat struct's `columns()` in
`include/utils/stats.h`, so adding a field there surfaces it in the CSV. Only the
**measured** search sweeps are logged to `queries.csv`; the progress searches run
while an update is still in flight are not.

`index_snapshots.csv` rows are emitted every `INDEX_SNAPSHOT_EVERY` steps (and on
the last step). The snapshot is **structural only** — out/in-degree per live node
and the live-slots-per-page histogram — computed on the live index via its
`loc2id` map (correct despite dynamic updates). It does **not** include C2
coverage (exact C2 is O(N²)); use the offline `tests/index_metrics` tool for C2
on a static index.

## File formats

- **Vectors / queries** (`.bin`): `int32 num_vectors | int32 dim | row-major values`.
- **Checkpoint GT** (`gt_<N>.bin`, DiskANN truthset): `int32 nq | int32 k | nq*k uint32 ids | nq*k float dists`.
  `N` is the number of vectors inserted so far (a multiple of `index_npts/STEP`).

## Generating ground truth (required for recall)

Exact GT per step is expensive, so PipeANN uses a shortcut: compute the top-1000
of the *full* dataset once, then pick each checkpoint's live top-k from it.

```bash
# 1. full-dataset GT (topk >= 1000):
build/tests/utils/compute_groundtruth <type> l2 DATA_BIN QUERY_FILE 1000 truth.bin null null
# 2. per-checkpoint GT for the sliding window (last arg 0 = insert+delete, 1 = insert-only):
build/tests/utils/gt_update truth.bin <index_npts> <total_npts> <batch=index_npts/STEP> 10 TRUTHSET_DIR 0
```

The confirmed-clean `gt_*.parquet` files under `fnct-db-eval/logs/` are **not**
usable here: different format, and insert-only (no deletes). This workload deletes,
so recall needs `.bin` GT regenerated for the sliding-window live sets.

## The configs

| Config | Status |
|--------|--------|
| `example_tiny.conf` | **Verified end-to-end** (4096×16 toy data). Artifacts on `/mnt/ssd4t/sarun/fastdann-exp/tiny`; regenerate with `make_example_data.py` + the commands in that file. |
| `sift_10m.conf` | Template. Datasets confirmed present; needs index build + GT (minutes). |
| `sift_100m.conf` / `spacev_100m.conf` | Template. Needs a 200M data slice, a 100M index build (~5h), and GT. |

## Build / environment notes

- The repo build target `overall_performance` required a one-line fix: it never
  called `index.start_bg_io()`, so with the committed `-DBG_IO_THREAD` optimization
  inserts deadlocked. Fixed in `tests/overall_performance.cpp`.
- The root filesystem was full, so this was built out-of-source on ssd4t:
  ```bash
  ln -sf $(ldconfig -p|awk '/libmkl_rt.so/{print $NF;exit}') /mnt/ssd4t/sarun/cblas_shim/libcblas.so
  cmake -S . -B /mnt/ssd4t/sarun/pipeann-build -DIO_ENGINE=aio
  LIBRARY_PATH=/mnt/ssd4t/sarun/cblas_shim TMPDIR=/mnt/ssd4t/sarun/tmp \
    make -C /mnt/ssd4t/sarun/pipeann-build -j overall_performance build_disk_index gt_update change_pts compute_groundtruth
  ```
  Point the runner at it with `PIPEANN_BIN=` and `LD_LIBRARY_PATH=/mnt/ssd4t/sarun/cblas_shim`.
