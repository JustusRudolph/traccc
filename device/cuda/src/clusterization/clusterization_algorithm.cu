/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// CUDA Library include(s).
#include "../utils/utils.hpp"
#include "traccc/cuda/clusterization/clusterization_algorithm.hpp"
#include "traccc/cuda/utils/barrier.hpp"
#include "traccc/cuda/utils/definitions.hpp"

// Project include(s)
#include "traccc/clusterization/device/aggregate_cluster.hpp"
#include "traccc/clusterization/device/ccl_kernel.hpp"
#include "traccc/clusterization/device/form_spacepoints.hpp"
#include "traccc/clusterization/device/reduce_problem_cell.hpp"

// Vecmem include(s).
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <algorithm>

namespace traccc::cuda {

namespace {
/// These indices in clusterization will only range from 0 to
/// max_cells_per_partition, so we only need a short.
using index_t = unsigned short;

static constexpr int TARGET_CELLS_PER_THREAD = 8;
static constexpr int MAX_CELLS_PER_THREAD = 12;
}  // namespace

namespace kernels {

// __global__ void find_clusters_cell_parallel(
//     const cell_container_types::const_view cells_view,
//     vecmem::data::vector_view<std::size_t> cell_to_module_view,
//     vecmem::data::vector_view<std::size_t> cell_indices_in_mod_view,
//     vecmem::data::jagged_vector_view<unsigned int> cell_cluster_label_view,
//     vecmem::data::vector_view<std::size_t> clusters_per_module_view) {
//     /*
//      * this function is the same as find_clusters but instead of every module
//      * being a thread, every cell is a thread instead. Thus, it has an
//      * extra argument which makes it possible to map the current cell (idx) to
//      * the module it belongs to.
//      */
//     unsigned int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
    
//     // Initialize the device container for cells
//     cell_container_types::const_device cells_device(cells_view);
//     // Do the same with cell to module and cell index mapping
//     vecmem::device_vector<std::size_t> device_cell_to_module(
//         cell_to_module_view);
//     vecmem::device_vector<std::size_t> device_cell_indices_in_mod(
//         cell_indices_in_mod_view);

//     // Ignore if idx is out of range
//     if (thread_idx >= device_cell_to_module.size())
//         return;

//     // get the current module number from the current cell idx
//     std::size_t module_number = device_cell_to_module.at(thread_idx);
//     std::size_t cell_index = device_cell_indices_in_mod.at(thread_idx);

//     // Initialise the jagged device vector for cell cluster indices
//     // and the device vector for the number of clusters per module
//     vecmem::jagged_device_vector<unsigned int> device_cell_cluster_labels(
//         cell_cluster_label_view);
//     vecmem::device_vector<std::size_t> device_clusters_per_module(
//         clusters_per_module_view);

//     // Get the cells for the current module and the cell this thread
//     // is looking at
//     const vecmem::device_vector<const traccc::cell>& cells =
//         cells_device.at(module_number).items;
//     // Get the relevant labels, so the ones for this current module
//     vecmem::device_vector<unsigned int> cluster_labels = 
//         device_cell_cluster_labels[module_number];
    
//     // find nearest neighbour above/left and write into current
//     unsigned int NN_index =
//         device::setup_cluster_labels_and_NN(cell_index, cells, cluster_labels);

//     if (NN_index == cells.size()) {
//         // we have hit an origin label, set it
//         std::size_t* cluster_size = &device_clusters_per_module[module_number];
//         // need to case, atomicAdd not overloaded for size_t
//         unsigned int* cluster_size_uint = (unsigned int*) cluster_size;
//         unsigned int cluster_label = atomicAdd(cluster_size_uint, 1) + 1;
        
//         cluster_labels[cell_index] = cluster_label;
//     }

//     // ensure all cells have found their NN before continuing:
//     __syncthreads();
//     // lastly, look through the labels and assign iteratively
//     device::fconn_find(cell_index, cluster_labels);
// }


/// CUDA kernel for running @c traccc::device::ccl_kernel
__global__ void ccl_kernel(
    const cell_collection_types::const_view cells_view,
    const cell_module_collection_types::const_view modules_view,
    const index_t max_cells_per_partition,
    const index_t target_cells_per_partition,
    measurement_collection_types::view measurements_view,
    unsigned int& measurement_count,
    vecmem::data::vector_view<unsigned int> cell_links) {
    __shared__ unsigned int partition_start, partition_end;
    __shared__ unsigned int outi;
    extern __shared__ index_t shared_v[];
    index_t* f = &shared_v[0];
    index_t* f_next = &shared_v[max_cells_per_partition];
    traccc::cuda::barrier barry_r;

    device::ccl_kernel(threadIdx.x, blockDim.x, blockIdx.x, cells_view,
                       modules_view, max_cells_per_partition,
                       target_cells_per_partition, partition_start,
                       partition_end, outi, f, f_next, barry_r,
                       measurements_view, measurement_count, cell_links);
}

__global__ void form_spacepoints(
    measurement_collection_types::const_view measurements_view,
    cell_module_collection_types::const_view modules_view,
    const unsigned int measurement_count,
    spacepoint_collection_types::view spacepoints_view) {

    device::form_spacepoints(threadIdx.x + blockIdx.x * blockDim.x,
                             measurements_view, modules_view, measurement_count,
                             spacepoints_view);
}

}  // namespace kernels

clusterization_algorithm::clusterization_algorithm(
    const traccc::memory_resource& mr, vecmem::copy& copy, stream& str,
    const unsigned short target_cells_per_partition)
    : m_mr(mr),
      m_copy(copy),
      m_stream(str),
      m_target_cells_per_partition(target_cells_per_partition) {}

clusterization_algorithm::output_type clusterization_algorithm::operator()(
    const cell_collection_types::const_view& cells,
    const cell_module_collection_types::const_view& modules) const {

    // Get a convenience variable for the stream that we'll be using.
    cudaStream_t stream = details::get_stream(m_stream);

    // Number of cells
    const cell_collection_types::view::size_type num_cells =
        m_copy.get_size(cells);

    if (num_cells == 0) {
        return {output_type::first_type{0, m_mr.main},
                output_type::second_type{0, m_mr.main}};
    }

    // Create result object for the CCL kernel with size overestimation
    measurement_collection_types::buffer measurements_buffer(num_cells,
                                                             m_mr.main);
    m_copy.setup(measurements_buffer);

    // Counter for number of measurements
    vecmem::unique_alloc_ptr<unsigned int> num_measurements_device =
        vecmem::make_unique_alloc<unsigned int>(m_mr.main);
    CUDA_ERROR_CHECK(cudaMemsetAsync(num_measurements_device.get(), 0,
                                     sizeof(unsigned int), stream));

    const unsigned short max_cells_per_partition =
        (m_target_cells_per_partition * MAX_CELLS_PER_THREAD +
         TARGET_CELLS_PER_THREAD - 1) /
        TARGET_CELLS_PER_THREAD;
    const unsigned int threads_per_partition =
        (m_target_cells_per_partition + TARGET_CELLS_PER_THREAD - 1) /
        TARGET_CELLS_PER_THREAD;
    const unsigned int num_partitions =
        (num_cells + m_target_cells_per_partition - 1) /
        m_target_cells_per_partition;

    // Create buffer for linking cells to their spacepoints.
    vecmem::data::vector_buffer<unsigned int> cell_links(num_cells, m_mr.main);
    m_copy.setup(cell_links);

    // Launch ccl kernel. Each thread will handle a single cell.
    kernels::
        ccl_kernel<<<num_partitions, threads_per_partition,
                     2 * max_cells_per_partition * sizeof(index_t), stream>>>(
            cells, modules, max_cells_per_partition,
            m_target_cells_per_partition, measurements_buffer,
            *num_measurements_device, cell_links);

    CUDA_ERROR_CHECK(cudaGetLastError());

    // Copy number of measurements to host
    vecmem::unique_alloc_ptr<unsigned int> num_measurements_host =
        vecmem::make_unique_alloc<unsigned int>(
            (m_mr.host != nullptr) ? *(m_mr.host) : m_mr.main);
    CUDA_ERROR_CHECK(cudaMemcpyAsync(
        num_measurements_host.get(), num_measurements_device.get(),
        sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
    m_stream.synchronize();

    spacepoint_collection_types::buffer spacepoints_buffer(
        *num_measurements_host, m_mr.main);
    m_copy.setup(spacepoints_buffer);

    // For the following kernel, we can now use whatever the desired number of
    // threads per block.
    auto spacepointsLocalSize = 1024;
    const unsigned int num_blocks =
        (*num_measurements_host + spacepointsLocalSize - 1) /
        spacepointsLocalSize;

    // Turn 2D measurements into 3D spacepoints
    kernels::form_spacepoints<<<num_blocks, spacepointsLocalSize, 0, stream>>>(
        measurements_buffer, modules, *num_measurements_host,
        spacepoints_buffer);

    CUDA_ERROR_CHECK(cudaGetLastError());

    return {std::move(spacepoints_buffer), std::move(cell_links)};
}

}  // namespace traccc::cuda