#!/usr/bin/env python3
"""Plot the static index metrics emitted by index_metrics (JSON) and the
per-query pages-accessed CSV from search_disk_index. One PNG per metric; each
snapshot in the JSON becomes an overlaid series, so the same script works once
the insert/delete workload emits multiple snapshots.

Usage: plot_metrics.py <metrics.json> <pages_per_search.csv|none> <out_dir> [visited_per_hop.csv|none]
"""
import json
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def hist(ax, series, title, xlabel, bins=40):
    for label, values in series:
        if not values:
            continue
        ax.hist(values, bins=bins, alpha=0.55, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.legend()


def main():
    metrics_json, pages_csv, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    snaps = json.load(open(metrics_json))["snapshots"]

    def series(key):
        return [(s["label"], s[key]) for s in snaps]

    # Outdegree / indegree / C2 coverage: one histogram figure each.
    for key, title, xlabel, fname in [
        ("outdeg", "Out-degree distribution (sampled)", "out-degree", "outdegree.png"),
        ("indeg", "In-degree distribution (sampled)", "in-degree", "indegree.png"),
        ("c2_coverage", "C2 short-hop coverage (sampled)", "fraction of true k-NN within 2 hops", "c2_coverage.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        hist(ax, series(key), title, xlabel)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/{fname}", dpi=150)
        plt.close(fig)

    # Entries per page: bar chart over occupancy 0..nnodes_per_sector.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s in snaps:
        pf = s["entries_per_page"]
        ax.bar(range(len(pf)), pf, alpha=0.55, label=s["label"])
    ax.set_title("Live records per page")
    ax.set_xlabel("records on page")
    ax.set_ylabel("# pages")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/entries_per_page.png", dpi=150)
    plt.close(fig)

    # # pages in disk index: one bar per snapshot (a trend once there are several).
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([s["label"] for s in snaps], [s["n_pages"] for s in snaps])
    ax.set_title("# pages in disk index")
    ax.set_ylabel("pages")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/n_pages.png", dpi=150)
    plt.close(fig)

    # Pages accessed per search, split by search L (from the CSV).
    if pages_csv != "none":
        by_L = defaultdict(list)
        with open(pages_csv) as f:
            for line in f:
                L, nio = line.split(",")
                by_L[int(L)].append(int(nio))
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for L in sorted(by_L):
            ax.hist(by_L[L], bins=40, alpha=0.5, label=f"L={L}")
        ax.set_title("Pages accessed per search")
        ax.set_xlabel("pages (n_ios)")
        ax.set_ylabel("count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{out_dir}/pages_per_search.png", dpi=150)
        plt.close(fig)

    # Per-round search progress (visited / min-distance / unnecessary), per L.
    # Per-hop averages carry each query forward at its final value after it
    # terminates (its frontier and best distance stop changing), so every hop
    # averages over ALL queries and the tail isn't biased toward slow queries.
    visited_csv = sys.argv[4] if len(sys.argv) > 4 else "none"
    if visited_csv != "none":
        # (L, query) -> {hop: (visited, min_dist, unnecessary, kth_dist)}
        curves = defaultdict(dict)
        with open(visited_csv) as f:
            for line in f:
                L, q, hop, v, mind, unnec, kth = line.split(",")
                curves[(int(L), int(q))][int(hop)] = (int(v), float(mind), int(unnec), float(kth))
        per_L = defaultdict(list)
        for (L, _q), hv in curves.items():
            per_L[L].append(hv)

        def carry(hv, hop, idx):
            """value of component idx at hop, carried forward past the query's end."""
            h = hop if hop in hv else max(k for k in hv if k <= hop)
            return hv[h][idx]

        def hop_curve(title, ylabel, idx, fname):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for L in sorted(per_L):
                qs = per_L[L]
                max_hop = max(h for hv in qs for h in hv)
                avg = [sum(carry(hv, hop, idx) for hv in qs) / len(qs) for hop in range(1, max_hop + 1)]
                ax.plot(range(1, max_hop + 1), avg, marker=".", label=f"L={L}")
            ax.set_title(title)
            ax.set_xlabel("expansion round (hop)")
            ax.set_ylabel(ylabel)
            ax.legend()
            fig.tight_layout()
            fig.savefig(f"{out_dir}/{fname}", dpi=150)
            plt.close(fig)

        hop_curve("Frontier growth: avg visited vs round", "avg # visited nodes", 0, "visited_per_hop.png")
        hop_curve("Top-1 (min) distance to query vs round", "avg best exact distance", 1, "min_dist_per_hop.png")
        hop_curve("k-th (10th) best distance to query vs round", "avg k-th best exact distance", 3,
                  "kth_dist_per_hop.png")

        # Per-search scalars: total visited and # unnecessary expansions -> histograms.
        def per_search_hist(title, xlabel, reduce_fn, fname):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for L in sorted(per_L):
                vals = [reduce_fn(hv) for hv in per_L[L]]
                ax.hist(vals, bins=40, alpha=0.5, label=f"L={L}")
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("count")
            ax.legend()
            fig.tight_layout()
            fig.savefig(f"{out_dir}/{fname}", dpi=150)
            plt.close(fig)

        per_search_hist("Total visited nodes per search", "# visited",
                        lambda hv: hv[max(hv)][0], "visited_per_search.png")
        per_search_hist("Unnecessary expansions per search", "# rounds adding no in-range candidate",
                        lambda hv: sum(c[2] for c in hv.values()), "unnecessary_per_search.png")

    print(f"wrote plots to {out_dir}/")


if __name__ == "__main__":
    main()
