/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// CUDA Library include(s).
#include "traccc/cuda/clusterization/clusterization_algorithm.hpp"

// Project include(s)
#include "traccc/clusterization/device/connect_components.hpp"
#include "traccc/clusterization/device/count_cluster_cells.hpp"
#include "traccc/clusterization/device/create_measurements.hpp"
#include "traccc/clusterization/device/find_clusters.hpp"
#include "traccc/clusterization/device/form_spacepoints.hpp"
#include "traccc/cuda/utils/make_prefix_sum_buff.hpp"
#include "traccc/device/fill_prefix_sum.hpp"

// Vecmem include(s).
#include <vecmem/utils/copy.hpp>
#include <vecmem/utils/cuda/copy.hpp>

// System include(s).
#include <algorithm>
#include <string>
#include <chrono>

// Local include(s)
#include "traccc/cuda/utils/definitions.hpp"

namespace traccc::cuda {
namespace kernels {


// __global__ void setup_cluster_labels(
//     vecmem::data::vector_view<std::size_t> cell_to_module_view,
//     vecmem::data::vector_view<std::size_t> cell_indices_in_mod_view,
//     vecmem::data::jagged_vector_view<unsigned int> cell_cluster_label_view) {
//     /*
//         * This function sets up labels for every cell. Before running
//         * Hoshen-Kopelman, each vector in the jagged vector cell_cluster_label_view
//         * (which corresponds to one module each), is changed from being all
//         * zeros to an array from 1 to N, where N is the number of cells in
//         * the given module that are activated. Note that because we are dealing
//         * with an inconsistent device vector, we just launch K threads, where K
//         * is the total number of cells. This is in place of doing a N threads
//         * for each module, the (module, cell) number being represented by (x,y)
//         * in the threadIdx.
//         */
//     // get global index
//     unsigned int global_idx = threadIdx.x + blockIdx.x * blockDim.x;

//     device::set_init_cluster_labels(global_idx, cell_to_module_view, 
//                                     cell_indices_in_mod_view,
//                                     cell_cluster_label_view);
// }


__global__ void find_clusters_cell_parallel(
    const cell_container_types::const_view cells_view,
    vecmem::data::vector_view<std::size_t> cell_to_module_view,
    vecmem::data::vector_view<std::size_t> cell_indices_in_mod_view,
    vecmem::data::jagged_vector_view<unsigned int> cell_cluster_label_view,
    vecmem::data::vector_view<std::size_t> clusters_per_module_view) {
    /*
     * this function is the same as find_clusters but instead of every module
     * being a thread, every cell is a thread instead. Thus, it has an
     * extra argument which makes it possible to map the current cell (idx) to
     * the module it belongs to.
     */
    unsigned int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Initialize the device container for cells
    cell_container_types::const_device cells_device(cells_view);
    // Do the same with cell to module and cell index mapping
    vecmem::device_vector<std::size_t> device_cell_to_module(
        cell_to_module_view);
    vecmem::device_vector<std::size_t> device_cell_indices_in_mod(
        cell_indices_in_mod_view);

    // Ignore if idx is out of range
    if (thread_idx >= device_cell_to_module.size())
        return;

    // get the current module number from the current cell idx
    std::size_t module_number = device_cell_to_module.at(thread_idx);
    std::size_t cell_index = device_cell_indices_in_mod.at(thread_idx);

    // Initialise the jagged device vector for cell cluster indices
    // and the device vector for the number of clusters per module
    vecmem::jagged_device_vector<unsigned int> device_cell_cluster_labels(
        cell_cluster_label_view);
    vecmem::device_vector<std::size_t> device_clusters_per_module(
        clusters_per_module_view);

    // Get the cells for the current module and the cell this thread
    // is looking at
    const vecmem::device_vector<const traccc::cell>& cells =
        cells_device.at(module_number).items;
    // Get the relevant labels, so the ones for this current module
    vecmem::device_vector<unsigned int> cluster_labels = 
        device_cell_cluster_labels[module_number];
    
    // bool label_changed = true;
    // unsigned int neighbour_index = 0;
    // neighbour_index--;  // wrap around to be max to ensure no init neighbour def
    // unsigned int n_iters = 0;

    // while(label_changed) {
    //     label_changed = device::find_clusters_cell_parallel_passthrough(
    //                     module_number, cell_index, cells,
    //                     cluster_labels, neighbour_index);
    //     __syncthreads();
    //     n_iters++;
    // }
    // // if (n_iters >= 10) {
    // //     printf("%d iterations in cell %d.\n", n_iters, cell_idx);
    // // }
    unsigned int NN_index =
        device::setup_cluster_labels_and_NN(cell_index, cells, cluster_labels);

    if (NN_index == cells.size()) {
        //printf("Found an origin cell!\n");
        // we have hit an origin label, set it
        std::size_t* cluster_size = &device_clusters_per_module[module_number];
        unsigned int* cluster_size_uint = (unsigned int*) cluster_size;
        unsigned int cluster_label = atomicAdd(cluster_size_uint, 1) + 1;
        // device::write_value(&cluster_labels[cell_index], cluster_label);
        cluster_labels[cell_index] = cluster_label;
    }

    // first write NN value into current
    //device::write_from_NN(cell_index, cells, cluster_labels);
    // ensure all cells have found their NN before continuing:
    __syncthreads();
    // lastly, look through the labels and assign iteratively
    device::hk_find(cell_index, cluster_labels);
}


__global__ void normalise_cluster_numbers(
    vecmem::data::jagged_vector_view<unsigned int> cell_cluster_label_view,
    vecmem::data::vector_view<std::size_t> clusters_per_module_view) {

    /*
     * This function sets the cluster labels to be 1 -> N, where N is the
     * number of clusters in the module. After the initial HK clusterisation,
     * there are still N unique labels only, but they are not necessarily
     * numbered 1 -> N. Furthermore, this function also writes the number of
     * labels in a module (so N) to the clusters_per_module vector.
     */
    unsigned int module_number = threadIdx.x + blockIdx.x * blockDim.x;
    device::normalise_cluster_numbers(module_number, cell_cluster_label_view,
                                      clusters_per_module_view);
}


__global__ void find_clusters(
    const cell_container_types::const_view cells_view,
    vecmem::data::jagged_vector_view<unsigned int> cell_cluster_label_view,
    vecmem::data::vector_view<std::size_t> clusters_per_module_view) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    device::find_clusters(idx, cells_view, cell_cluster_label_view,
                          clusters_per_module_view);
}


__global__ void print_debug_module(
    const cell_container_types::const_view cells_view,
    vecmem::data::jagged_vector_view<unsigned int> cell_cluster_label_view,
    vecmem::data::vector_view<std::size_t> clusters_per_module_view,
    bool check_all, bool before_clust, unsigned int module_to_check) {
    /*
     * This is a debug function which prints information about the current
     * module. Current module is the global threadIdx.
     */

    cell_container_types::const_device cells_device(cells_view);
    vecmem::jagged_device_vector<unsigned int> device_cell_cluster_label(
        cell_cluster_label_view);
    vecmem::device_vector<std::size_t> device_clusters_per_module(
        clusters_per_module_view);

    // 755 has the most cells
    // 2464 has exactly one cluster with the most cells (5)
    // module_to_check = 755;
    
    int module_number = threadIdx.x + blockIdx.x * blockDim.x;
    // if all modules are to be checked, then check if index is in range
    // otherwise, check only if index is right
    bool check = (check_all && (module_number < cells_device.size())) ||
                 (module_number == module_to_check);
    if (check && before_clust) {
        // first print the cells for this module
        const vecmem::device_vector<const traccc::cell>& cells =
            cells_device.at(module_number).items;
        const traccc::cell_module cells_header =
            cells_device.at(module_number).header;
        size_t n_cells = cells.size();

        // TODO figure out what "module" means here
        printf("cells_header: id: %d, module: %d range0: (%d, %d), range1: (%d, %d)\n",
               (int) cells_header.event, (int) cells_header.module, 
               (int) cells_header.range0[0], (int) cells_header.range0[1],
               (int) cells_header.range1[0], (int) cells_header.range1[1]);

        for (int i=0; i < n_cells; i++) {
            const traccc::cell cell = cells[i];
            scalar act = cell.activation;
            printf("Idx: %d, i.e. (%d, %d): Cell index %d and position: (%d, %d) activation: %f\n",
                    module_number, blockIdx.x, threadIdx.x, i, cell.channel0, cell.channel1, act);

        }
        // then print the inputs/outputs that go into the actual algorithm

        // THE BELOW IS COMMENTED BECAUSE IT'S NOT REALLY NECESSARY, ALL CELLS ARE ALLOCATED
        // TO CLUSTER NUMBER 0 AT INITIALISATION

        // unsigned int n_cell_cluster_label = device_cell_cluster_label.at(idx).size();
        // for (int i=0; i < n_cell_cluster_label; i++) {
        //     unsigned int cell_cluster_number = device_cell_cluster_label.at(idx).at(i);
        //     printf("Before: Idx: %d, Clusters in module: %d, Cell %d belongs to cluster %d\n",
        //         idx, (int) device_clusters_per_module.at(idx), i, cell_cluster_number);
        // }

        printf("Before: Idx: %d, Clusters in module: %d. All cells are in cluster 0.\n",
            module_number, (int) device_clusters_per_module.at(module_number));
    }

    if (check && !before_clust) {
        const vecmem::device_vector<const traccc::cell>& cells =
            cells_device.at(module_number).items;
        unsigned int n_cell_cluster_label =
            device_cell_cluster_label.at(module_number).size();
        //printf("n_cell_cluster_label: %u\n", n_cell_cluster_label);
        // print outputs from clusterisation algo
        // printf("After: %d clusters in module %d.\n",
        //        (int) device_clusters_per_module.at(module_number), (int) module_number);

        for (int i=0; i < n_cell_cluster_label; i++) {
            const traccc::cell cell = cells[i];
            unsigned int cell_cluster_number =
                device_cell_cluster_label.at(module_number).at(i);
            printf("After: Module: %d, Clusters in module: %d, Cell %d, Point (%d, %d) belongs to cluster %d. Activation: %f\n",
                module_number, (int) device_clusters_per_module.at(module_number), i, cell.channel0,
                cell.channel1, cell_cluster_number, cell.activation);
        }
    }
    // if (device_clusters_per_module.at(idx) == 2) {
    //     printf("Module with two clusters: %d. Number of cells: %d\n",
    //            idx, cells_device.at(idx).items.size());
    // }
    //printf("Module %d has %d clusters\n.", idx, device_clusters_per_module.at(idx));
}

__global__ void count_cluster_cells(
    vecmem::data::jagged_vector_view<unsigned int> cell_cluster_label_view,
    vecmem::data::vector_view<std::size_t> cluster_prefix_sum_view,
    vecmem::data::vector_view<const device::prefix_sum_element_t>
        cells_prefix_sum_view,
    vecmem::data::vector_view<unsigned int> cluster_sizes_view) {

    device::count_cluster_cells(
        threadIdx.x + blockIdx.x * blockDim.x, cell_cluster_label_view,
        cluster_prefix_sum_view, cells_prefix_sum_view, cluster_sizes_view);
}

__global__ void connect_components(
    const cell_container_types::const_view cells_view,
    vecmem::data::jagged_vector_view<unsigned int> cell_cluster_label_view,
    vecmem::data::vector_view<std::size_t> cluster_prefix_sum_view,
    vecmem::data::vector_view<const device::prefix_sum_element_t>
        cells_prefix_sum_view,
    cluster_container_types::view clusters_view) {

    device::connect_components(threadIdx.x + blockIdx.x * blockDim.x,
                               cells_view, cell_cluster_label_view,
                               cluster_prefix_sum_view, cells_prefix_sum_view,
                               clusters_view);
}
__global__ void create_measurements(
    const cell_container_types::const_view cells_view,
    cluster_container_types::const_view clusters_view,
    measurement_container_types::view measurements_view) {

    device::create_measurements(threadIdx.x + blockIdx.x * blockDim.x,
                                clusters_view, cells_view, measurements_view);
}

__global__ void form_spacepoints(
    measurement_container_types::const_view measurements_view,
    vecmem::data::vector_view<const device::prefix_sum_element_t>
        measurements_prefix_sum_view,
    spacepoint_container_types::view spacepoints_view) {

    device::form_spacepoints(threadIdx.x + blockIdx.x * blockDim.x,
                             measurements_view, measurements_prefix_sum_view,
                             spacepoints_view);
}

}  // namespace kernels

clusterization_algorithm::clusterization_algorithm(
    const traccc::memory_resource& mr)
    : m_mr(mr) {

    // Initialize m_copy ptr based on memory resources that were given
    if (mr.host) {
        m_copy = std::make_unique<vecmem::cuda::copy>();
    } else {
        m_copy = std::make_unique<vecmem::copy>();
    }
}

clusterization_algorithm::output_type clusterization_algorithm::operator()(
    const cell_container_types::host& cells_per_event) const {

    // Vecmem copy object for moving the data between host and device
    vecmem::copy copy;

    printf("Starting CUDA clusterization.\n");
    // Number of modules
    unsigned int num_modules = cells_per_event.size();
    printf("Number of modules: %d\n", num_modules);

    // Work block size for kernel execution
    std::size_t threadsPerBlock = 64;
    std::size_t blocksPerGrid;  // initialise, will change dep. on kernel
    // for debug: Choose which level of parallelisation
    bool parallelise_by_cell = true;

    // Get the view of the cells container
    auto cells_data =
        get_data(cells_per_event, (m_mr.host ? m_mr.host : &(m_mr.main)));

    //int test = (int) cells_data;
    // auto *headers = &cells_data.headers;
    // auto *cells = vecmem::get_data(&cells_data.items);
    // for (unsigned int j = 0; j < num_modules; j++){
    //     auto module_cells = cells[j];
    //     std::sort(module_cells->begin(), module_cells->end(),
    //         [](auto a, auto b)
    //             {
    //                 return a.channel1 > b.channel1;
    //             });
    // }

    // Get the sizes of the cells in each module
    auto cell_sizes = copy.get_sizes(cells_data.items);
    unsigned int n_cells_total = 0;
    for (unsigned int i = 0; i < cell_sizes.size(); i++) {
        n_cells_total += cell_sizes[i];
    }
    // create a vector which maps the cell index to current module
    vecmem::vector<std::size_t> cell_to_module(n_cells_total);
    vecmem::vector<std::size_t> cell_indices_in_module(n_cells_total);
    unsigned int curr_idx = 0;  // used to populate the above
    for (std::size_t i = 0; i < cell_sizes.size(); i++) {
        // if (i == 3000) {
        //         printf("Module 3000 initialise: \n\n");
        //     }
        for (std::size_t j = 0; j < cell_sizes[i]; j++) {
            cell_to_module[curr_idx] = i;  // i is the module number
            // j is the cell number in the module
            cell_indices_in_module[curr_idx] = j;
            // if (i == 3000) {
            //     printf("cell_indices_in_module[%d] = %d\n", (int) curr_idx, (int) j);
            // }
            curr_idx++;
        }
    }
    // instantiate vector buffers to hold the above data
    vecmem::data::vector_buffer<std::size_t> cell_to_module_buff(
        n_cells_total, m_mr.main);
    m_copy->setup(cell_to_module_buff);
    
    vecmem::data::vector_buffer<std::size_t> cell_indices_in_mod_buff(
        n_cells_total, m_mr.main);
    m_copy->setup(cell_indices_in_mod_buff);
    
    // move the vectors to the device and create vector views
    (*m_copy)(vecmem::get_data(cell_to_module), cell_to_module_buff,
        vecmem::copy::type::copy_type::host_to_device);
    (*m_copy)(vecmem::get_data(cell_indices_in_module), cell_indices_in_mod_buff,
        vecmem::copy::type::copy_type::host_to_device);

    // create and move the vectors on the device to vector views
    vecmem::data::vector_view<std::size_t> cell_to_module_view =
        cell_to_module_buff;
    vecmem::data::vector_view<std::size_t> cell_indices_in_mod_view =
        cell_indices_in_mod_buff;
    /*
     * Helper container for clusterisation calculations.
     * Each inner vector corresponds to 1 module.
     * The indices in a particular inner vector will be filled by the
     * relevant clusterisation algorithm (sparse CCL or Hoshen-Kopelman)
     * and will indicate to which cluster, a particular cell in the module
     * belongs to.
     */
    vecmem::data::jagged_vector_buffer<unsigned int> cell_cluster_label_buff(
        std::vector<std::size_t>(cell_sizes.begin(), cell_sizes.end()),
        m_mr.main, m_mr.host);
    m_copy->setup(cell_cluster_label_buff);
    // Create view to pass to kernel which sets up cluster labels
    vecmem::data::jagged_vector_view<unsigned int> cell_cluster_label_view =
        cell_cluster_label_buff;

    // if (parallelise_by_cell) {
    //     // only if parallelising the clusterisation cell by cell do we
    //     // give each cell a label before clusterisation
    //     blocksPerGrid = (n_cells_total + threadsPerBlock - 1) / threadsPerBlock;

    //     // Invoke find clusters that will call cluster finding kernel
    //     kernels::setup_cluster_labels<<<blocksPerGrid, threadsPerBlock>>>(
    //         cell_to_module_view, cell_indices_in_mod_view, cell_cluster_label_view);

    //     CUDA_ERROR_CHECK(cudaGetLastError());
    //     CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    // }

    /*
     * cl_per_module_prefix_buff is a vector buffer with numbers of found
     * clusters in each module. Later it will be transformed into prefix sum
     * vector (hence the name). The logic is the following. After
     * cluster_finding_kernel, the buffer will contain cluster sizes e.i.
     *
     * cluster sizes: | 1 | 12 | 5 | 102 | 42 | ... - cl_per_module_prefix_buff
     * module index:  | 0 |  1 | 2 |  3  |  4 | ...
     *
     * Now, we copy those cluster sizes to the host and make a duplicate vector
     * of them. So, we are left with cl_per_module_prefix_host, and
     * clusters_per_module_host - which are the same. Now, we procede to
     * modifying the cl_per_module_prefix_host to actually resemble its name
     * i.e.
     *
     * We do std::inclusive_scan on it, which will result in a prefix sum
     * vector:
     *
     * cl_per_module_prefix_host: | 1 | 13 | 18 | 120 | 162 | ...
     *
     * Then, we copy this vector into the previous cl_per_module_prefix_buff.
     * In this way, we don't need to allocate the memory on the device twice.
     *
     * Now, the monotonic prefix sum buffer - cl_per_module_prefix_buff, will
     * allow us to insert the clusters at the correct position inside the
     * kernel. The remaining host vector - clusters_per_module_host, will be
     * needed to allocate memory for other buffers later in the code.
     */
    vecmem::data::vector_buffer<std::size_t> cl_per_module_prefix_buff(
        num_modules, m_mr.main);
    m_copy->setup(cl_per_module_prefix_buff);

    // Create views to pass to cluster finding kernel
    const cell_container_types::const_view cells_view(cells_data);
    vecmem::data::vector_view<std::size_t> cl_per_module_prefix_view =
        cl_per_module_prefix_buff;

    if (parallelise_by_cell) {
        // get the grid size for using all cells
        threadsPerBlock = 1024;
        blocksPerGrid = (n_cells_total + threadsPerBlock - 1) / threadsPerBlock;

        auto start_clusterisation_time = std::chrono::system_clock::now();
        // Run cell parallelised kernel to get clusters
        kernels::find_clusters_cell_parallel<<<blocksPerGrid, threadsPerBlock>>>(
            cells_view, cell_to_module_view, cell_indices_in_mod_view,
            cell_cluster_label_view, cl_per_module_prefix_view);

        CUDA_ERROR_CHECK(cudaGetLastError());
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());

        auto end_clusterisation_time = std::chrono::system_clock::now();

        std::chrono::duration<double> clusterisation_time =
            end_clusterisation_time - start_clusterisation_time;
        
        printf("TIME TAKEN FOR HK CLUSTERISATION: %fs\n", clusterisation_time.count());

        // go back to module wide parallelisation
        threadsPerBlock = 64;
        // blocksPerGrid = (num_modules + threadsPerBlock - 1) / threadsPerBlock;
        // kernels::print_debug_module<<<blocksPerGrid, threadsPerBlock>>>(
        //     cells_view, cell_cluster_label_view, cl_per_module_prefix_view,
        //     true, false, 3000);  // check for module 755 after clusterisation
        
        // CUDA_ERROR_CHECK(cudaGetLastError());
        // CUDA_ERROR_CHECK(cudaDeviceSynchronize());

        // normalise the output to have labels from just 1 -> N
        // kernels::normalise_cluster_numbers<<<blocksPerGrid, threadsPerBlock>>>(
        //     cell_cluster_label_view, cl_per_module_prefix_view);

        // CUDA_ERROR_CHECK(cudaGetLastError());
        // CUDA_ERROR_CHECK(cudaDeviceSynchronize());
        
        // auto end_normalisation_time = std::chrono::system_clock::now();
        // std::chrono::duration<double> normalisation_time =
        //     end_normalisation_time - end_clusterisation_time;
        
        // printf("TIME TAKEN FOR LABEL NORMALISATION: %fs\n", normalisation_time.count());
        // kernels::print_debug_module<<<blocksPerGrid, threadsPerBlock>>>(
        //     cells_view, cell_cluster_label_view, cl_per_module_prefix_view,
        //     false, false, 970);  // check for module 755 after clusterisation
        
        // CUDA_ERROR_CHECK(cudaGetLastError());
        // CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    }
    else {  // parallelise by module
        // Calculating grid size for cluster finding kernel
        blocksPerGrid = (num_modules + threadsPerBlock - 1) / threadsPerBlock;

        auto start_clusterisation_time = std::chrono::system_clock::now();
        // Invoke find clusters that will call cluster finding kernel
        kernels::find_clusters<<<blocksPerGrid, threadsPerBlock>>>(
            cells_view, cell_cluster_label_view, cl_per_module_prefix_view);

        CUDA_ERROR_CHECK(cudaGetLastError());
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());

        auto end_clusterisation_time = std::chrono::system_clock::now();

        std::chrono::duration<double> clusterisation_time =
            end_clusterisation_time - start_clusterisation_time;
        
        printf("TIME TAKEN FOR CCL CLUSTERISATION: %fs\n", clusterisation_time.count());
        
        // kernels::print_debug_module<<<blocksPerGrid, threadsPerBlock>>>(
        //     cells_view, cell_cluster_label_view, cl_per_module_prefix_view,
        //     false, false, 755);  // check for module 755 after clusterisation
        
        CUDA_ERROR_CHECK(cudaGetLastError());
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    }


    // Create prefix sum buffer
    vecmem::data::vector_buffer cells_prefix_sum_buff =
        make_prefix_sum_buff(cell_sizes, *m_copy, m_mr);

    // Copy the sizes of clusters per module to the host
    // and create a copy of "clusters per module" vector
    vecmem::vector<std::size_t> cl_per_module_prefix_host(
        m_mr.host ? m_mr.host : &(m_mr.main));
    (*m_copy)(cl_per_module_prefix_buff, cl_per_module_prefix_host,
              vecmem::copy::type::copy_type::device_to_host);
    std::vector<std::size_t> clusters_per_module_host(
        cl_per_module_prefix_host.begin(), cl_per_module_prefix_host.end());

    // Perform the inclusive scan operation
    std::inclusive_scan(cl_per_module_prefix_host.begin(),
                        cl_per_module_prefix_host.end(),
                        cl_per_module_prefix_host.begin());

    unsigned int total_clusters = cl_per_module_prefix_host.back();

    // Copy the prefix sum back to its device container
    (*m_copy)(vecmem::get_data(cl_per_module_prefix_host),
              cl_per_module_prefix_buff,
              vecmem::copy::type::copy_type::host_to_device);

    // Vector of the exact cluster sizes, will be filled in cluster counting
    vecmem::data::vector_buffer<unsigned int> cluster_sizes_buffer(
        total_clusters, m_mr.main);
    m_copy->setup(cluster_sizes_buffer);
    m_copy->memset(cluster_sizes_buffer, 0);

    // Create views to pass to cluster counting kernel
    vecmem::data::vector_view<const device::prefix_sum_element_t>
        cells_prefix_sum_view = cells_prefix_sum_buff;
    vecmem::data::vector_view<unsigned int> cluster_sizes_view =
        cluster_sizes_buffer;

    // Calclating grid size for cluster counting kernel (block size 64)
    blocksPerGrid =
        (cells_prefix_sum_view.size() + threadsPerBlock - 1) / threadsPerBlock;
    // Invoke cluster counting will call count cluster cells kernel
    kernels::count_cluster_cells<<<blocksPerGrid, threadsPerBlock>>>(
        cell_cluster_label_view, cl_per_module_prefix_view,
        cells_prefix_sum_view, cluster_sizes_view);
    // Check for kernel launch errors and Wait for the cluster_counting kernel
    // to finish
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // Copy cluster sizes back to the host
    std::vector<unsigned int> cluster_sizes;
    (*m_copy)(cluster_sizes_buffer, cluster_sizes,
              vecmem::copy::type::copy_type::device_to_host);

    // Cluster container buffer for the clusters and headers (cluster ids)
    cluster_container_types::buffer clusters_buffer{
        {total_clusters, m_mr.main},
        {std::vector<std::size_t>(total_clusters, 0),
         std::vector<std::size_t>(cluster_sizes.begin(), cluster_sizes.end()),
         m_mr.main, m_mr.host}};
    m_copy->setup(clusters_buffer.headers);
    m_copy->setup(clusters_buffer.items);

    // Create views to pass to component connection kernel
    cluster_container_types::view clusters_view = clusters_buffer;

    // Using previous block size and thread size (64)
    // Invoke connect components will call connect components kernel
    kernels::connect_components<<<blocksPerGrid, threadsPerBlock>>>(
        cells_view, cell_cluster_label_view, cl_per_module_prefix_view,
        cells_prefix_sum_view, clusters_view);
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // Resizable buffer for the measurements
    measurement_container_types::buffer measurements_buffer{
        {num_modules, m_mr.main},
        {std::vector<std::size_t>(num_modules, 0), clusters_per_module_host,
         m_mr.main, m_mr.host}};
    m_copy->setup(measurements_buffer.headers);
    m_copy->setup(measurements_buffer.items);

    // Spacepoint container buffer to fill inside the spacepoint formation
    // kernel
    spacepoint_container_types::buffer spacepoints_buffer{
        {num_modules, m_mr.main},
        {std::vector<std::size_t>(num_modules, 0), clusters_per_module_host,
         m_mr.main, m_mr.host}};
    m_copy->setup(spacepoints_buffer.headers);
    m_copy->setup(spacepoints_buffer.items);

    // Create views to pass to measurement creation kernel
    measurement_container_types::view measurements_view = measurements_buffer;

    // Calculating grid size for measurements creation kernel (block size 64)
    blocksPerGrid =
        (clusters_view.headers.size() - 1 + threadsPerBlock) / threadsPerBlock;

    // Invoke measurements creation will call create measurements kernel
    kernels::create_measurements<<<blocksPerGrid, threadsPerBlock>>>(
        cells_view, clusters_view, measurements_view);

    // Check for kernel launch errors and Wait here for the measurements
    // creation kernel to finish
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // Create prefix sum buffer
    vecmem::data::vector_buffer meas_prefix_sum_buff = make_prefix_sum_buff(
        m_copy->get_sizes(measurements_buffer.items), *m_copy, m_mr);

    // Create views to run spacepoint formation
    vecmem::data::vector_view<const device::prefix_sum_element_t>
        meas_prefix_sum_view = meas_prefix_sum_buff;
    spacepoint_container_types::view spacepoints_view = spacepoints_buffer;

    // Using the same grid size as before
    // Invoke spacepoint formation will call form_spacepoints kernel
    kernels::form_spacepoints<<<blocksPerGrid, threadsPerBlock>>>(
        measurements_view, meas_prefix_sum_view, spacepoints_view);
    // Check for kernel launch errors and Wait for the spacepoint formation
    // kernel to finish
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    return spacepoints_buffer;
}

}  // namespace traccc::cuda