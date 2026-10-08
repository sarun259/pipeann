#!/usr/bin/env bash
# PipeANN SIFT-10M insert+delete sliding-window experiment on PSC Bridges-2 RM.
#
#   sbatch tests/psc_bridges2_10m.sbatch
#
# RM = Regular Memory node: 128 cores, 256 GB RAM, ~3.84 TB node-local NVMe at
# $LOCAL. Bridges-2 runs RHEL8 (no io_uring) so PipeANN must be built with
# -DIO_ENGINE=aio (see the one-time PREP below). The experiment is I/O-heavy and
# must run on $LOCAL (node-local SSD), NOT on Ocean (/ocean, shared Lustre):
# this script stages inputs to $LOCAL, runs, then copies results back to Ocean.
#
# ============================= ONE-TIME PREP =================================
# Do this once on a compute node (use `interact -p RM -t 2:00:00`; login nodes
# forbid heavy compute). $PROJECT is your Ocean project dir.
#
#   module load gcc cmake mkl            # confirm names: `module avail gcc cmake mkl`
#   # cblas shim: src/CMakeLists.txt links bare -lcblas; point it at MKL's libmkl_rt:
#   mkdir -p $PROJECT/cblas_shim
#   ln -sf "$MKLROOT/lib/intel64/libmkl_rt.so" $PROJECT/cblas_shim/libcblas.so
#   # Build (aio engine; RHEL8 has no io_uring):
#   cd $PROJECT/FastDANN/pipeann
#   LIBRARY_PATH=$PROJECT/cblas_shim cmake -S . -B build -DIO_ENGINE=aio
#   LIBRARY_PATH=$PROJECT/cblas_shim cmake --build build -j --target \
#       overall_performance build_disk_index compute_groundtruth gt_update change_pts
#
#   # Inputs on Ocean (stage sift.100M.128.u8bin + sift.10K.128.u8.query here first):
#   IN=$PROJECT/fastdann-exp/sift10m ; mkdir -p "$IN/gt" ; cd "$IN"
#   # 20M header'd slice (10M seed + 10M to insert):
#   $PROJECT/FastDANN/pipeann/build/tests/utils/change_pts uint8 sift.100M.128.u8bin 20000000
#   mv sift.100M.128.u8bin20000000 sift.20M.128.u8bin
#   # 10M header'd slice to build the index from:
#   $PROJECT/FastDANN/pipeann/build/tests/utils/change_pts uint8 sift.100M.128.u8bin 10000000
#   mv sift.100M.128.u8bin10000000 sift.10M.128.u8bin
#   # Build the 10M update-capable index (R L PQ_bytes M T; tune per README):
#   $PROJECT/FastDANN/pipeann/build/tests/build_disk_index \
#       uint8 sift.10M.128.u8bin index 64 128 32 1 128 l2 pq
#   # Ground truth for the sliding window (topk>=1000 over the 20M, then per-checkpoint):
#   $PROJECT/FastDANN/pipeann/build/tests/utils/compute_groundtruth \
#       uint8 l2 sift.20M.128.u8bin sift.10K.128.u8.query 1000 truth.bin null null
#   $PROJECT/FastDANN/pipeann/build/tests/utils/gt_update \
#       truth.bin 10000000 20000000 100000 10 gt 0      # 100000 = 10M/STEP, 0 = insert+delete
# ============================================================================

#SBATCH -J pipeann_10m
#SBATCH -p RM                   # full Regular-Memory node (128 cores, node-local NVMe)
#SBATCH -N 1
#SBATCH --ntasks-per-node=1     # single process; parallelism is OpenMP threads inside it
#SBATCH -t 08:00:00             # wall clock — adjust to your run
#SBATCH -A XXXXXXX              # <-- YOUR PSC grant/allocation id (see `projects`)
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
set -euo pipefail

# ---- Paths you control (edit) ----------------------------------------------
REPO=$PROJECT/FastDANN/pipeann          # the checkout on Ocean (persistent)
BUILD=$REPO/build                        # built once in PREP; holds tests/overall_performance
INPUTS=$PROJECT/fastdann-exp/sift10m     # prepared index*, 20M bin, query, gt/ (from PREP)
SHIM=$PROJECT/cblas_shim                 # libcblas.so -> MKL libmkl_rt.so (from PREP)

# ---- Stage inputs to node-local NVMe (fast; wiped when the job ends) --------
WORK=$LOCAL/sift10m
mkdir -p "$WORK"
echo "[$(date +%T)] staging inputs -> $WORK on $(hostname)"
cp -v "$INPUTS"/index*                "$WORK"/     # _disk.index(.tags), _pq_*, _mem*, ...
cp -v "$INPUTS"/sift.20M.128.u8bin    "$WORK"/
cp -v "$INPUTS"/sift.10K.128.u8.query "$WORK"/
cp -rv "$INPUTS"/gt                   "$WORK"/

# ---- Environment -----------------------------------------------------------
module load gcc mkl || true              # BLAS at runtime; confirm module names
export LD_LIBRARY_PATH=$SHIM:${LD_LIBRARY_PATH:-}
export PIPEANN_BIN=$BUILD/tests/overall_performance

# ---- Run (STATS_DIR / INDEX_SNAPSHOT_EVERY come from the config) ------------
cd "$REPO"
echo "[$(date +%T)] starting experiment"
LOG=$WORK/run.log ./run_experiment.sh configs/sift_10m_psc.conf

# ---- Save results back to Ocean (node-local scratch is wiped at job end) ----
RESULTS=$INPUTS/results.$SLURM_JOB_ID
mkdir -p "$RESULTS"
cp -rv "$WORK"/stats "$WORK"/run.log "$RESULTS"/ 2>/dev/null || true
echo "[$(date +%T)] done. results -> $RESULTS"
