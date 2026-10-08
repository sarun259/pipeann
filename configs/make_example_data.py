#!/usr/bin/env python3
"""Generate the tiny example dataset + queries for example_tiny.conf.

Writes two files in PipeANN's .bin format (used by every data/query file):

    | int32 num_vectors | int32 dim | num_vectors * dim values (row-major) |

The example is deliberately small so the whole insert+delete pipeline
(build index -> generate ground truth -> run) finishes in seconds. See
configs/README.md for how these feed into the sliding-window workload.
"""
import argparse
import numpy as np

# dtype string -> (numpy dtype, PipeANN type name for the binaries)
DTYPES = {"uint8": (np.uint8, "uint8"), "int8": (np.int8, "int8"), "float": (np.float32, "float")}


def write_bin(path, arr):
    n, d = arr.shape
    with open(path, "wb") as f:
        np.array([n, d], dtype=np.int32).tofile(f)
        arr.tofile(f)
    print(f"wrote {path}: {n} x {d} ({arr.dtype})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="directory to write data.bin and queries.bin")
    ap.add_argument("--n", type=int, default=4096, help="total vectors in data.bin")
    ap.add_argument("--nq", type=int, default=64, help="number of queries")
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--dtype", choices=list(DTYPES), default="uint8")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np_dtype, _ = DTYPES[args.dtype]
    rng = np.random.default_rng(args.seed)
    # uint8/int8 -> small integer vectors; float -> standard normal.
    if np.issubdtype(np_dtype, np.integer):
        info = np.iinfo(np_dtype)
        data = rng.integers(0, 256, size=(args.n, args.dim)).astype(np_dtype)
        queries = rng.integers(0, 256, size=(args.nq, args.dim)).astype(np_dtype)
        data = np.clip(data, info.min, info.max).astype(np_dtype)
        queries = np.clip(queries, info.min, info.max).astype(np_dtype)
    else:
        data = rng.standard_normal((args.n, args.dim)).astype(np_dtype)
        queries = rng.standard_normal((args.nq, args.dim)).astype(np_dtype)

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    write_bin(os.path.join(args.out_dir, "data.bin"), data)
    write_bin(os.path.join(args.out_dir, "queries.bin"), queries)


if __name__ == "__main__":
    main()
