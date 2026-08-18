// sssp_csr.cu
//
// Gunrock SSSP that reads a CSR graph from stdin:
//
//   ./a.out < graph.txt
//
// graph.txt format:
//   line 1: E V source
//   line 2: E column-indices (neighbor list)
//   line 3: E weights        (one per edge, parallel to column-indices)
//   line 4: row-offsets — either V values or V+1 values.
//           If only V values are given, E is appended as the
//           final offset.
//
// Example (matches the format you posted):
//   6 5 0
//   1 2 2 3 4 4
//   1 1 3 2 1 3
//   0 2 3 5 6
//
// -------------------------------------------------------------
// Build (against a built Gunrock "essentials" checkout):
//
//   nvcc -std=c++17 -O3 \
//       -I<gunrock_root>/include \
//       -I<gunrock_root>/externals/thrust \
//       -I<gunrock_root>/externals/cub \
//       -I<gunrock_root>/externals/modern-gpu/src \
//       sssp_csr.cu -o a.out
//
// Or drop this file into gunrock/examples/algorithms/sssp/ and
// add it to that directory's CMakeLists.txt as its own target;
// then `cmake --build build --target sssp_csr` will pick up all
// the right include paths automatically.
// -------------------------------------------------------------

#include <iostream>
#include <vector>
#include <string>
#include <ctime>
#include <limits>
#include <cstdio>

#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/algorithms/sssp.hxx>

using namespace gunrock;
using namespace memory;

void run_sssp() {
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

  std::vector<weight_t> h_values(num_edges);
  for (edge_t i = 0; i < num_edges; i++) {
    std::cin >> h_values[i];
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
  // Run SSSP
  // ---------------------------------------------------------------
  vertex_t n_vertices = G.get_number_of_vertices();

  thrust::device_vector<weight_t> distances(n_vertices);
  thrust::device_vector<vertex_t> predecessors(n_vertices);

  clock_t starttt = clock();
  gunrock::sssp::run(G, source, distances.data().get(),
                      predecessors.data().get());
  clock_t enddd = clock();

  // ---------------------------------------------------------------
  // Output
  // ---------------------------------------------------------------
  thrust::host_vector<weight_t> h_distances = distances;

  weight_t maxi = 0;
  weight_t unreachable = std::numeric_limits<weight_t>::max();
  for (vertex_t v = 0; v < n_vertices; v++) {
    if (h_distances[v] != unreachable && h_distances[v] > maxi) {
      maxi = h_distances[v];
    }
  }

  std::cout << std::endl;
  std::cout << "#########SSSP GUNROCK ORIGINAL########" << std::endl;
  std::cout << "vertices           : " << num_vertices << '\n';
  std::cout << "edges              : " << num_edges << '\n';
  std::cout << "source             : " << source << std::endl;
  std::cout << "Max Distance       : " << maxi << std::endl;
  printf("Time Taken (GPU)   : %f ms\n",
         (((double)enddd - (double)starttt) / CLOCKS_PER_SEC) * 1000);
}

int main(int argc, char** argv) {
  run_sssp();
  return 0;
}