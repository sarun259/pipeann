// Static graph-structure metrics for a prebuilt PipeANN disk index.
//
// Reads the on-disk graph directly (assumes the equal id<->loc mapping of a
// static / freshly-converted index: loc i holds node id i, all locs live) and
// dumps ONE JSON "snapshot" of distributions for tests/plot_metrics.py. Node
// coords are taken from the index itself, so C2 is self-consistent and needs no
// base file. The JSON is an array of snapshots so the (future) insert/delete
// workload can emit one per sampling period and reuse the plots.
//
// Accumulated / insert-time metrics (allocation-rule frequency, back-edges
// added-vs-pruned) are produced by the workload, not here. "# pages accessed
// per search" comes from search_disk_index (PAGES_PER_SEARCH_OUT).
//
// Usage: index_metrics <float|int8|uint8> <index_prefix> <l2> <sample_size>
//                      <k> <out.json> [seed]

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include <omp.h>

#include "ssd_index.h"            // SSDIndexMetadata, DiskNode, SECTOR_LEN
#include "utils.h"                // load_bin
#include "utils/stats.h"          // IndexStats
#include "utils/tsl/robin_set.h"

namespace {
  // Exact squared-L2 in the native element type (promoted to double).
  template<typename T>
  double l2sq(const T *a, const T *b, uint64_t dim) {
    double s = 0;
    for (uint64_t i = 0; i < dim; ++i) {
      double d = (double) a[i] - (double) b[i];
      s += d * d;
    }
    return s;
  }

  // CSR adjacency + per-node coords, built in one sequential pass of the graph.
  // Coords come from the index itself (indexed by internal node id), so C2 is
  // self-consistent and needs no base file or id<->tag mapping.
  template<typename T>
  struct Graph {
    uint64_t npoints = 0, nnodes_per_sector = 0, n_pages = 0, dim = 0;
    std::vector<uint64_t> offset;      // size npoints+1, CSR row pointers (into nbrs)
    std::vector<uint32_t> nbrs;        // flat in-range neighbor ids, size == kept edges
    std::vector<uint32_t> outdeg_all;  // size npoints, true out-degree (node.nnbrs)
    std::vector<uint32_t> indeg;       // size npoints, full in-degree
    std::vector<uint32_t> page_fill;   // histogram[0..nnodes_per_sector]
    std::vector<T> coords;             // size npoints*dim, node vectors
  };

  template<typename T>
  Graph<T> scan_graph(const std::string &disk_index_file, const pipeann::SSDIndexMetadata<T> &meta) {
    Graph<T> g;
    g.npoints = meta.npoints;
    g.nnodes_per_sector = meta.nnodes_per_sector;
    g.dim = meta.data_dim;
    g.n_pages = (meta.npoints + meta.nnodes_per_sector - 1) / meta.nnodes_per_sector;
    g.offset.assign(meta.npoints + 1, 0);
    g.outdeg_all.assign(meta.npoints, 0);
    g.indeg.assign(meta.npoints, 0);
    g.page_fill.assign(meta.nnodes_per_sector + 1, 0);
    g.coords.assign(meta.npoints * meta.data_dim, (T) 0);

    std::ifstream in(disk_index_file, std::ios::binary);
    if (!in) {
      LOG(ERROR) << "cannot open " << disk_index_file;
      exit(-1);
    }
    in.seekg(SECTOR_LEN, std::ios::beg);  // data sectors follow the metadata sector

    // 64MB reads; each data sector holds nnodes_per_sector consecutive locs.
    constexpr uint64_t kChunkSectors = 16384;
    std::vector<char> buf(kChunkSectors * SECTOR_LEN);
    uint64_t loc = 0;
    for (uint64_t page = 0; page < g.n_pages; page += kChunkSectors) {
      uint64_t pages = std::min(kChunkSectors, g.n_pages - page);
      in.read(buf.data(), pages * SECTOR_LEN);
      for (uint64_t p = 0; p < pages && loc < g.npoints; ++p) {
        char *sector = buf.data() + p * SECTOR_LEN;
        uint64_t live = 0;
        for (uint64_t s = 0; s < g.nnodes_per_sector && loc < g.npoints; ++s, ++loc) {
          // loc == id under equal mapping; DiskNode finds the slot within the page.
          pipeann::DiskNode<T> node(sector, (uint32_t) loc, meta);
          std::memcpy(g.coords.data() + loc * g.dim, node.coords, g.dim * sizeof(T));
          uint32_t nn = node.nnbrs;
          g.outdeg_all[loc] = nn;
          g.offset[loc] = g.nbrs.size();  // CSR row start = current flat size
          for (uint32_t m = 0; m < nn; ++m) {
            uint32_t id = node.nbrs[m];
            if (id < g.npoints) {  // guard against stray ids; none expected for a static index
              g.nbrs.push_back(id);
              g.indeg[id]++;
            }
          }
          live++;
        }
        g.page_fill[live]++;
      }
    }
    g.offset[g.npoints] = g.nbrs.size();
    return g;
  }

  // exact k nearest ids to `center` over the whole base (self excluded).
  template<typename T>
  std::vector<uint32_t> exact_knn(uint32_t center, const T *base, uint64_t npoints, uint64_t dim, uint64_t k) {
    std::vector<std::pair<double, uint32_t>> heap;  // max-heap of (dist, id), size <= k
    heap.reserve(k + 1);
    const T *q = base + center * dim;
    for (uint64_t j = 0; j < npoints; ++j) {
      if (j == center) {
        continue;
      }
      double d = l2sq(q, base + j * dim, dim);
      if (heap.size() < k) {
        heap.emplace_back(d, (uint32_t) j);
        std::push_heap(heap.begin(), heap.end());
      } else if (d < heap.front().first) {
        std::pop_heap(heap.begin(), heap.end());
        heap.back() = {d, (uint32_t) j};
        std::push_heap(heap.begin(), heap.end());
      }
    }
    std::vector<uint32_t> ids;
    ids.reserve(heap.size());
    for (auto &h : heap) {
      ids.push_back(h.second);
    }
    return ids;
  }

  void write_u32_array(std::ostream &o, const std::vector<uint32_t> &v) {
    o << "[";
    for (size_t i = 0; i < v.size(); ++i) {
      o << (i ? "," : "") << v[i];
    }
    o << "]";
  }
  void write_f64_array(std::ostream &o, const std::vector<double> &v) {
    o << "[";
    for (size_t i = 0; i < v.size(); ++i) {
      o << (i ? "," : "") << v[i];
    }
    o << "]";
  }

  // Serialize snapshots as {"snapshots":[{...},...]} for tests/plot_metrics.py.
  // An array so the (future) insert/delete workload can emit one IndexStats per
  // sampling period and reuse the same plotting path.
  void write_snapshots(std::ostream &o, const std::vector<pipeann::IndexStats> &snaps) {
    o << "{\"snapshots\":[";
    for (size_t i = 0; i < snaps.size(); ++i) {
      const pipeann::IndexStats &s = snaps[i];
      o << (i ? "," : "") << "{";
      o << "\"label\":\"" << s.label << "\",";
      o << "\"n_live\":" << s.n_live << ",";
      o << "\"n_pages\":" << s.n_pages << ",";
      o << "\"nnodes_per_sector\":" << s.nnodes_per_sector << ",";
      o << "\"outdeg\":";
      write_u32_array(o, s.outdeg);
      o << ",\"indeg\":";
      write_u32_array(o, s.indeg);
      o << ",\"entries_per_page\":";
      write_u32_array(o, s.entries_per_page);
      o << ",\"c2_coverage\":";
      write_f64_array(o, s.c2_coverage);
      o << "}";
    }
    o << "]}\n";
  }

  template<typename T>
  int run(int argc, char **argv) {
    const std::string index_prefix = argv[2];
    // argv[3] metric reserved; only L2 is implemented (matches the uint8 SIFT indices).
    const uint64_t sample_size = std::stoull(argv[4]);
    const uint64_t k = std::stoull(argv[5]);
    const std::string out_json = argv[6];
    const uint32_t seed = argc > 7 ? (uint32_t) std::stoul(argv[7]) : 1234;

    pipeann::SSDIndexMetadata<T> meta;
    meta.load_from_disk_index(index_prefix + "_disk.index");
    const uint64_t dim = meta.data_dim;

    LOG(INFO) << "scanning graph: " << meta.npoints << " nodes, degree<=" << meta.range;
    Graph<T> g = scan_graph<T>(index_prefix + "_disk.index", meta);

    // Sample nodes (shared by outdegree / indegree / C2).
    std::mt19937 rng(seed);
    const uint64_t S = std::min<uint64_t>(sample_size, g.npoints);
    std::vector<uint32_t> sample(g.npoints);
    for (uint64_t i = 0; i < g.npoints; ++i) {
      sample[i] = (uint32_t) i;
    }
    std::shuffle(sample.begin(), sample.end(), rng);
    sample.resize(S);

    std::vector<uint32_t> outdeg(S), indeg(S);
    for (uint64_t i = 0; i < S; ++i) {
      outdeg[i] = g.outdeg_all[sample[i]];
      indeg[i] = g.indeg[sample[i]];
    }

    // C2 short-hop coverage: fraction of a node's true k-NN (in the index's own
    // coord space) that lie within 2 hops in the graph.
    LOG(INFO) << "computing C2 over " << S << " sampled nodes (k=" << k << ")";
    const T *base = g.coords.data();

    std::vector<double> c2(S);
#pragma omp parallel for schedule(dynamic, 16)
    for (int64_t i = 0; i < (int64_t) S; ++i) {
      uint32_t u = sample[i];
      // 2-hop reachable set: 1-hop neighbors and their neighbors.
      tsl::robin_set<uint32_t> two_hop;
      for (uint64_t e = g.offset[u]; e < g.offset[u + 1]; ++e) {
        uint32_t v = g.nbrs[e];
        two_hop.insert(v);
        for (uint64_t f = g.offset[v]; f < g.offset[v + 1]; ++f) {
          two_hop.insert(g.nbrs[f]);
        }
      }
      std::vector<uint32_t> knn = exact_knn<T>(u, base, g.npoints, dim, k);
      uint32_t hit = 0;
      for (uint32_t id : knn) {
        if (two_hop.count(id)) {
          hit++;
        }
      }
      c2[i] = knn.empty() ? 0.0 : (double) hit / (double) knn.size();
    }

    pipeann::IndexStats snap;
    snap.label = "static";
    snap.n_live = g.npoints;
    snap.n_pages = g.n_pages;
    snap.nnodes_per_sector = g.nnodes_per_sector;
    snap.outdeg = std::move(outdeg);
    snap.indeg = std::move(indeg);
    snap.entries_per_page = g.page_fill;
    snap.c2_coverage = std::move(c2);

    std::ofstream o(out_json);
    write_snapshots(o, {snap});
    LOG(INFO) << "wrote " << out_json;
    return 0;
  }
}  // namespace

int main(int argc, char **argv) {
  if (argc < 7) {
    std::cout << "Usage: " << argv[0]
              << " <float|int8|uint8> <index_prefix> <l2> <sample_size> <k> <out.json> [seed]" << std::endl;
    return -1;
  }
  const std::string t = argv[1];
  if (t == "float") {
    return run<float>(argc, argv);
  } else if (t == "int8") {
    return run<int8_t>(argc, argv);
  } else if (t == "uint8") {
    return run<uint8_t>(argc, argv);
  }
  std::cout << "unsupported type " << t << std::endl;
  return -1;
}
