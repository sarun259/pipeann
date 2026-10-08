# PipeANN — project context

PipeANN is a graph-based, on-SSD, updatable vector store (billion-scale ANN).
See `README.md` for features/benchmarks. This file captures the non-obvious
context for *working in this checkout on this server* — read it before building,
running, or touching the index internals.

## What we're doing here

This is a fork used for an **index-metrics / evaluation** project: measuring
static graph-structure properties and per-search dynamics of PipeANN indices,
building toward an **insert+delete workload on 10M** with "sampling periods"
(snapshots at 10%, 20%, … of inserts) that emit distribution metrics. Target
metrics (OdinANN-style): degree distributions, C2 short-hop coverage,
pages/search, entries/page, #pages, allocation-rule frequency, back-edges
added-vs-pruned. Static ones are done; the accumulated/insert-time ones await
the workload. See `## Metrics harness`.

## Build & run on this server (important)

The umbrella (fnct-db) build is gone; this is a **standalone** build with three
quirks (all already handled by `tests/run_static_search.sh`):

- **No io_uring** → configure `cmake -S . -B build -DIO_ENGINE=aio`.
- **No standalone `libcblas.so`** (only MKL's `libmkl_rt`, which exports CBLAS).
  `src/CMakeLists.txt` links bare `-lcblas`, so symlink a shim and put it on
  `LIBRARY_PATH` for the link step (the script does this at `/tmp/cblas_shim`).
- **`tests/` is guarded by `PROJECT_IS_TOP_LEVEL`** and `tests/CMakeLists.txt`
  does **not** `add_subdirectory(utils)` by default (the umbrella configured it)
  — the working tree adds it so `compute_groundtruth` builds. If a clean
  checkout breaks, re-apply these.
- When you add a CMake target, run `cmake` as its own step before `make`;
  letting `make` trigger the reconfigure mid-run leaves the new target
  unresolved until the next invocation.

**One-liner for everything (build + GT + search + recall + metrics + plots):**
`bash tests/run_static_search.sh [sift1m|sift100m]`. Scratch/caches live in
`/tmp/pipeann_static`; plots go to `plots/<dataset>/`.

## Environment facts

- **Prebuilt indices are in `/mnt/ssd4t/tyang425/*`** (e.g. `pipeann/idx` = 1M
  SIFT, uint8/dim128/L2/PQ). They are owned by `tyang425`, mode 644, and we are
  **not** in the `shared` group → read-only to us.
- The reader (`linux_aligned_file_reader.cpp` `open()`) now honors
  `enable_writes`: a read-only search opens `O_RDONLY`, so prebuilt indices are
  read **in place, no copy**. (It used to force `O_RDWR`, which failed with
  EACCES → `fd=-1` → later surfaced as `io_submit` EBADF, because release builds
  compile out the `assert`.)
- **No writable multi-GB space.** `/` (holds `/tmp` and `/home`) has ~6 GB free;
  the 439 GB on `/mnt/ssd4t` is not writable to us. So we can read any index in
  place but cannot build/convert a large one (a 10M index is ~4 GB, a 100M one
  ~44 GB) without space from tyang.
- **No usable 100M PipeANN index yet:** `fnct-db-eval/index/pipeann002_sift` is a
  dirtied dynamic snapshot (graph references ids up to ~2.25e8 with only 100M PQ
  entries and no saved id→loc map → static search segfaults). The parlay
  `*_100m.flat/.graph` are ingestable via `load_graph`/`save_from_graph` but need
  ~44 GB to write. Datasets (uint8 SIFT base/query, GT) live under
  `/mnt/hdd2t/tyang425/fnct-db-eval/datasets` and `.../fnct-db-eval/index`.
- Python has **no pip/numpy/matplotlib**, but `python3 -m venv` + pip works and
  the network is up; plots run from a cached venv at `/tmp/pipeann_static/venv`.

## Index internals worth knowing (non-obvious)

- **id vs loc vs tag.** Internal node id == physical loc under the "equal
  mapping" (used when no `_partition.bin.aligned` exists). The external **tag**
  (`_disk.index.tags`) is separate — base/GT files are ordered by **tag**, not by
  internal id. The search returns tags; recall compares tags vs GT base ids.
  Pitfall: analyzing the graph by id and indexing a base file by that id is
  wrong (it bit the C2 metric). Use the index's own stored coords instead.
- **On-disk layout:** sector 0 is metadata; data sectors pack
  `nnodes_per_sector` nodes of `max_node_len` bytes each (`SECTOR_LEN`=4096).
  `SSDIndexMetadata::load_from_disk_index` + `DiskNode<T>(page, loc, meta)` parse
  it; neighbor ids are stored in id-space. Distances are squared-L2; PQ is
  approximate (traversal), coords give exact distance on expanded nodes.
- **Stats structs** live in `include/utils/stats.h` (renamed from
  `percentile_stats.h`). `QueryStats`, `InsertionStats`, `DeletionStats` share a
  CRTP base `Stats<Derived>` (static `mean`/`percentile`/`interval_means`/`to_csv`
  reductions over a sample array via a scalar accessor + a `columns()` each);
  `IndexStats` is standalone (a disk-scan snapshot of distribution vectors).
  `QueryStats` is threaded through the search/insert kernels; search modes beam /
  page / pipe / coro — the per-round search-progress fields are only filled by
  **beam** (`src/search/beam_search.cpp`, `do_beam_search`). `InsertionStats`
  (filled in `direct_insert.cpp`) embeds the insert's internal-search `QueryStats`.
- **Config-driven experiment** (`run_experiment.sh` + `configs/`) runs the
  `overall_performance` sliding-window insert+delete workload. With `STATS_DIR`
  set it emits raw per-event CSVs (insertions/queries/deletions) whose stat
  columns come from each struct's `columns()`; `INDEX_SNAPSHOT_EVERY=N` dumps a
  structural `IndexStats` snapshot (`index_snapshots.csv`) every N steps via
  `SSDIndex::compute_index_stats` (live `loc2id` walk, no C2). See `configs/README.md`.
- **Allocation (`SSDIndex::alloc_loc`, `src/ssd_index.cpp`)** has three rules in
  priority order (the OdinANN rules): (1) empty pages from the `empty_pages`
  queue (freed by merges), (2) hint pages touched by this insert's search if a
  page has ≥`threshold` holes, (3) new pages appended at `cur_loc`.
- **Dynamic updates** are in-place; inserts commit on a background thread
  (`start_bg_io()`/`stop_bg_io()` around a C++ insert loop). Deletes are lazy;
  `merge_deletes` compacts and is what feeds allocation rule 1.

## Metrics harness (added for this project)

- `tests/index_metrics.cpp` — one-pass graph scanner → JSON "snapshot" (array of
  snapshots, so the insert workload can append). Covers outdegree, indegree, C2
  coverage, entries/page, #pages. Assumes equal id↔loc (static index).
- `tests/search_disk_index.cpp` — env-gated per-query dumps:
  `PAGES_PER_SEARCH_OUT` (pages/search) and `VISITED_PER_HOP_OUT`
  (`L,query,hop,visited,min_dist,unnecessary,kth_dist` per expansion round).
- `tests/plot_metrics.py` — matplotlib (venv) → PNGs in `plots/<dataset>/`.
- `tests/run_static_search.sh` — orchestrates all of it; `sift100m` branch is
  scaffolding, guarded behind `ALLOW_DIRTY_100M=1`.

## Conventions

Global prefs apply (clear > clever, fail loud, comment only nonstandard logic,
keep diffs small, progress output for long runs). Project-specific: do **not**
commit or alter git history unless asked; `plots/` and `tests/run_static_search.sh`
are untracked working artifacts; core-library edits (the `O_RDONLY` reader fix,
`QueryStats` fields, `alloc_loc` instrumentation) stay in the working tree,
uncommitted, unless told otherwise.
