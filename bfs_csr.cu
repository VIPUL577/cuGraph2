// bfs_csr.cu
//
// Gunrock BFS that reads a CSR graph from stdin:
//
//   ./a.out < graph.txt
//
// graph.txt format:
//   line 1: E V source
//   line 2: E column-indices (neighbor/edge list)
//   line 3: row-offsets — either V values or V+1 values.
//           If only V values are given, E is appended as the
//           final offset (as in your example, where the last
//           offset E=20 is implied rather than written).
//
// Example (matches the format you posted):
//   20 10 0
//   1 2 5 0 3 3 4 5 6 7 2 6 8 7 9 8 9 0 1 4
//   0 3 5 7 10 11 13 15 16 18
//
// -------------------------------------------------------------
// Build (against a built Gunrock "essentials" checkout):
//
//   nvcc -std=c++17 -O3 \
//       -I<gunrock_root>/include \
//       -I<gunrock_root>/externals/thrust \
//       -I<gunrock_root>/externals/cub \
//       -I<gunrock_root>/externals/modern-gpu/src \
//       bfs_csr.cu -o a.out
//
// Or drop this file into gunrock/examples/algorithms/bfs/ and
// add it to that directory's CMakeLists.txt as its own target;
// then `cmake --build build --target bfs_csr` will pick up all
// the right include paths automatically.
// -------------------------------------------------------------

#include <iostream>
#include <vector>
#include <string>
#include <ctime>
#include <limits>
#include <cstdio>

#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/algorithms/bfs.hxx>

using namespace gunrock;
using namespace memory;

void run_bfs() {
  using vertex_t = int;
  using edge_t   = int;
  using weight_t = float;

  // ---------------------------------------------------------------
  // Read CSR graph from stdin
  // ---------------------------------------------------------------
  edge_t   num_edges;
  vertex_t num_vertices;
  vertex_t source;

  std::cin >> num_edges >> num_vertices >> source;

  std::vector<vertex_t> h_col_indices(num_edges);
  for (edge_t i = 0; i < num_edges; i++) {
    std::cin >> h_col_indices[i];
  }

  std::vector<edge_t> h_row_offsets;
  edge_t val;
  while (std::cin >> val) {
    h_row_offsets.push_back(val);
  }

  if ((vertex_t)h_row_offsets.size() == num_vertices) {
    // last offset (== E) was omitted; append it
    h_row_offsets.push_back(num_edges);
  } else if ((vertex_t)h_row_offsets.size() != num_vertices + 1) {
    std::cerr << "Error: row_offsets has " << h_row_offsets.size()
              << " entries, expected " << num_vertices << " or "
              << (num_vertices + 1) << std::endl;
    exit(1);
  }

  std::vector<weight_t> h_values(num_edges, 1.0f);  // unweighted -> all 1s

  // ---------------------------------------------------------------
  // Build CSR (host) and copy to device
  // ---------------------------------------------------------------
  format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t> csr;
  csr.number_of_rows     = num_vertices;
  csr.number_of_columns  = num_vertices;
  csr.number_of_nonzeros = num_edges;

  csr.row_offsets    = h_row_offsets;    // host -> device (thrust)
  csr.column_indices = h_col_indices;
  csr.nonzero_values  = h_values;

  auto G = graph::build::from_csr<memory_space_t::device, graph::view_t::csr>(
      csr.number_of_rows,
      csr.number_of_columns,
      csr.number_of_nonzeros,
      csr.row_offsets.data().get(),
      csr.column_indices.data().get(),
      csr.nonzero_values.data().get());

  // ---------------------------------------------------------------
  // Run BFS
  // ---------------------------------------------------------------
  vertex_t n_vertices = G.get_number_of_vertices();

  thrust::device_vector<vertex_t> distances(n_vertices);
  thrust::device_vector<vertex_t> predecessors(n_vertices);

  clock_t starttt = clock();
  gunrock::bfs::run(G, source, distances.data().get(),
                     predecessors.data().get());
  clock_t enddd = clock();

  // ---------------------------------------------------------------
  // Output
  // ---------------------------------------------------------------
  thrust::host_vector<vertex_t> h_distances = distances;

  vertex_t maxi = 0;
  vertex_t unreachable = std::numeric_limits<vertex_t>::max();
  for (vertex_t v = 0; v < n_vertices; v++) {
    if (h_distances[v] != unreachable && h_distances[v] > maxi) {
      maxi = h_distances[v];
    }
  }

  std::cout << std::endl;
  std::cout << "#########BFS GUNROCK ORIGINAL########" << std::endl;
  std::cout << "vertices           : " << num_vertices << '\n';
  std::cout << "edges              : " << num_edges << '\n';
  std::cout << "source             : " << source << std::endl;
  std::cout << "Max Distance       : " << maxi << std::endl;
  printf("Time Taken (GPU)   : %f ms\n",
         (((double)enddd - (double)starttt) / CLOCKS_PER_SEC) * 1000);
}

int main(int argc, char** argv) {
  run_bfs();
  return 0;
}
