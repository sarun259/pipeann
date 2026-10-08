#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "utils.h"

namespace pipeann {
  // Stat structs. The three event-scoped ones (QueryStats, InsertionStats,
  // DeletionStats) are per-event samples: callers collect an *array* of them
  // (one per event) and reduce across it, so they share the Stats<> base below.
  // IndexStats is different -- a single disk-scan snapshot holding distribution
  // vectors -- so it is a standalone struct (reduce *within* it, not across an
  // array) and serializes itself as JSON snapshots in tests/index_metrics.cpp.
  //   QueryStats     -- one search      (src/search/*, tests/search_disk_index.cpp)
  //   InsertionStats -- one insert       (src/update/direct_insert.cpp)
  //   DeletionStats  -- one delete-merge (src/update/delete_merge.cpp)
  //   IndexStats     -- one disk scan    (tests/index_metrics.cpp)

  // CRTP base for per-event sample structs. All members are static helpers that
  // reduce over an array of `Derived` samples via a scalar accessor (the fields
  // are heterogeneous, so aggregation is always "pick a scalar, then reduce").
  // CRTP (not a virtual base) keeps `Derived` a plain struct with no vtable
  // pointer -- QueryStats lives in million-entry arrays on the search hot path.
  template<typename Derived>
  struct Stats {
    using Accessor = std::function<double(const Derived &)>;
    using Column = std::pair<std::string, Accessor>;  // CSV column: header + extractor

    // Extract one scalar column across all samples.
    static std::vector<double> column(const Derived *s, uint64_t len, const Accessor &f) {
      std::vector<double> v(len);
      for (uint64_t i = 0; i < len; i++) {
        v[i] = f(s[i]);
      }
      return v;
    }

    static double mean(const std::vector<double> &v) {
      if (v.empty()) {
        return 0;
      }
      double sum = 0;
      for (double x : v) {
        sum += x;
      }
      return sum / (double) v.size();
    }

    // `v` taken by value: percentile sorts a private copy, leaving the caller's untouched.
    static double percentile(std::vector<double> v, float p) {
      if (v.empty()) {
        return 0;
      }
      std::sort(v.begin(), v.end());
      uint64_t idx = (uint64_t) (p * (float) v.size());
      return v[idx >= v.size() ? v.size() - 1 : idx];
    }

    // Split [0,len) into `n_intervals` contiguous chunks, return the mean of each.
    // This is the "sampling periods" reduction: e.g. mean insert latency over
    // each 10% slice of an insert run, with samples kept in insertion order.
    static std::vector<double> interval_means(const std::vector<double> &v, uint64_t n_intervals) {
      std::vector<double> out;
      if (n_intervals == 0) {
        return out;
      }
      out.reserve(n_intervals);
      for (uint64_t i = 0; i < n_intervals; i++) {
        uint64_t lo = v.size() * i / n_intervals;        // chunk bounds chosen so every
        uint64_t hi = v.size() * (i + 1) / n_intervals;  // sample lands in exactly one chunk
        double sum = 0;
        for (uint64_t j = lo; j < hi; j++) {
          sum += v[j];
        }
        out.push_back(hi > lo ? sum / (double) (hi - lo) : 0.0);
      }
      return out;
    }

    // Convenience overloads that pull the column from a sample array, then reduce.
    static double mean(const Derived *s, uint64_t len, const Accessor &f) { return mean(column(s, len, f)); }
    static double percentile(const Derived *s, uint64_t len, float p, const Accessor &f) {
      return percentile(column(s, len, f), p);
    }
    static std::vector<double> interval_means(const Derived *s, uint64_t len, uint64_t n, const Accessor &f) {
      return interval_means(column(s, len, f), n);
    }

    // Sum the accessor over the first `tot` samples but divide by `cnt`. Used for
    // filter metrics where a per-query value should be averaged over a subset
    // (cnt) of the processed (tot) queries. cnt==0 -> 0.
    static double mean(const Derived *s, uint64_t cnt, uint64_t tot, const Accessor &f) {
      if (cnt == 0) {
        return 0;
      }
      double sum = 0;
      for (uint64_t i = 0; i < tot; i++) {
        sum += f(s[i]);
      }
      return sum / (double) cnt;
    }

    // Plotting feed: one CSV row per sample over `Derived`'s plottable scalar
    // columns (header first). tests/plot_metrics.py consumes these.
    static void to_csv(std::ostream &o, const Derived *s, uint64_t len) {
      const std::vector<Column> cols = Derived::columns();
      for (size_t c = 0; c < cols.size(); c++) {
        o << (c ? "," : "") << cols[c].first;
      }
      o << "\n";
      for (uint64_t i = 0; i < len; i++) {
        for (size_t c = 0; c < cols.size(); c++) {
          o << (c ? "," : "") << cols[c].second(s[i]);
        }
        o << "\n";
      }
    }
  };

  // Per-search metrics, threaded through the search kernels (beam/page/pipe/coro).
  // The per-hop progress vectors are only filled by beam search (do_beam_search).
  struct QueryStats : Stats<QueryStats> {
    double total_us = 0;        // total time to process query in micros
    double n_ios = 0;           // total # of IOs issued
    double io_us = 0;           // total time spent in IO
    double io_us1 = 0;          // total time spent in IO
    double head_us = 0;         // total time spent in in-memory index
    double cpu_us = 0;          // total time spent in CPU
    double cpu_us1 = 0;         // total time spent in CPU
    double cpu_us2 = 0;         // total time spent in CPU
    double n_cmps = 0;          // # cmps
    double n_exact = 0;         // # full-precision distances computed over fetched coords
    double n_hops = 0;          // # search hops
    double n_current_used = 0;  // # force return for latency limit
    // Per-expansion-round search progress (beam search), one entry per hop:
    std::vector<uint32_t> visited_per_hop;    // cumulative distinct nodes seen
    std::vector<float> min_dist_per_hop;      // best (top-1) exact distance to query so far
    std::vector<float> kth_dist_per_hop;      // k-th best exact distance so far (pairs with recall@k)
    std::vector<uint8_t> unnecessary_per_hop; // 1 if the round added no new in-range candidate
    double n_filter[N_FILTER_TYPES] = {0};
    double n_est_filter_reads[N_FILTER_TYPES] = {0};
    double n_est_filter_cmps[N_FILTER_TYPES] = {0};
    double n_filter_reads[N_FILTER_TYPES] = {0};
    double n_filter_accessed_vectors[N_FILTER_TYPES] = {0};
    double n_filter_false_positives[N_FILTER_TYPES] = {0};

    double filter_io_us[N_FILTER_TYPES] = {0};
    double filter_cpu_us[N_FILTER_TYPES] = {0};
    double filter_io_us1[N_FILTER_TYPES] = {0};

    static std::vector<Column> columns() {
      return {{"total_us", [](const QueryStats &s) { return s.total_us; }},
              {"n_ios", [](const QueryStats &s) { return s.n_ios; }},
              {"n_cmps", [](const QueryStats &s) { return s.n_cmps; }},
              {"n_exact", [](const QueryStats &s) { return s.n_exact; }},
              {"n_hops", [](const QueryStats &s) { return s.n_hops; }}};
    }
  };

  // Per-insert metrics. An insert first runs an internal search (filling `search`)
  // and then mutates the graph in place (the counters below). Filled by
  // SSDIndex::insert_in_place.
  struct InsertionStats : Stats<InsertionStats> {
    QueryStats search;          // stats of this insert's internal do_pipe_search
    double n_rmw_reads = 0;     // # pages this insert's graph-mutation read-modify-write asked to re-read
    double n_pages_touched = 0; // # distinct disk pages touched by this insert's graph-mutation read-modify-write
    double n_out_edges = 0;     // # outgoing edges written for the newly-inserted node
    double n_in_edges = 0;      // # existing neighbors whose edge list gained a back-edge to the new node
    double n_evictions = 0;     // # of those back-edge updates that evicted a pre-existing edge

    static std::vector<Column> columns() {
      return {{"n_rmw_reads", [](const InsertionStats &s) { return s.n_rmw_reads; }},
              {"n_pages_touched", [](const InsertionStats &s) { return s.n_pages_touched; }},
              {"n_out_edges", [](const InsertionStats &s) { return s.n_out_edges; }},
              {"n_in_edges", [](const InsertionStats &s) { return s.n_in_edges; }},
              {"n_evictions", [](const InsertionStats &s) { return s.n_evictions; }},
              {"search_total_us", [](const InsertionStats &s) { return s.search.total_us; }}};
    }
  };

  // Per-delete-merge metrics. Filled by SSDIndex::merge_deletes (which compacts
  // the index and feeds allocation rule 1). NOTE: no counters are wired yet --
  // the insert+delete workload will populate these; see merge_deletes TODOs.
  struct DeletionStats : Stats<DeletionStats> {
    double n_nodes_deleted = 0;     // # tags compacted out in this merge
    double n_pages_freed = 0;       // # pages returned to the empty_pages queue
    double n_back_edges_pruned = 0; // # neighbor edges pointing at a deleted node that were removed
    double n_pages_rewritten = 0;   // # pages rewritten during compaction

    static std::vector<Column> columns() {
      return {{"n_nodes_deleted", [](const DeletionStats &s) { return s.n_nodes_deleted; }},
              {"n_pages_freed", [](const DeletionStats &s) { return s.n_pages_freed; }},
              {"n_back_edges_pruned", [](const DeletionStats &s) { return s.n_back_edges_pruned; }},
              {"n_pages_rewritten", [](const DeletionStats &s) { return s.n_pages_rewritten; }}};
    }
  };

  // Static graph-structure metrics, one per sampling-period "snapshot", computed
  // by scanning a disk index (no live event). Filled by tests/index_metrics.cpp.
  struct IndexStats {
    std::string label = "static";            // sampling-period label (e.g. "static", "10%")
    uint64_t n_live = 0;                      // # live nodes (npoints for a static index)
    uint64_t n_pages = 0;                     // # data pages (sectors) on disk
    uint64_t nnodes_per_sector = 0;           // node slots per page
    std::vector<uint32_t> outdeg;             // out-degree per sampled node
    std::vector<uint32_t> indeg;              // in-degree per sampled node
    std::vector<uint32_t> entries_per_page;   // histogram[0..nnodes_per_sector] of page fill
    std::vector<double> c2_coverage;          // C2 short-hop coverage per sampled node
  };

}  // namespace pipeann
