/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::device {

TRACCC_HOST_DEVICE
void find_clusters(
    std::size_t globalIndex, const cell_container_types::const_view& cells_view,
    vecmem::data::jagged_vector_view<unsigned int> sparse_ccl_indices_view,
    vecmem::data::vector_view<std::size_t> clusters_per_module_view) {

    // Initialize the device container for cells
    cell_container_types::const_device cells_device(cells_view);

    // Ignore if idx is out of range
    if (globalIndex >= cells_device.size())
        return;

    // Get the cells for the current module
    const auto& cells = cells_device.at(globalIndex).items;

    // Vectors used for cluster indices found by sparse CCL
    vecmem::jagged_device_vector<unsigned int> device_sparse_ccl_indices(
        sparse_ccl_indices_view);
    auto cluster_indices = device_sparse_ccl_indices[globalIndex];

    // Run the sparse CCL algorithm
    unsigned int n_clusters = detail::sparse_ccl(cells, cluster_indices);
    // unsigned int n_clusters = detail::hoshen_kopelman(globalIndex, cells, cluster_indices);

    // Fill the "number of clusters per module" vector
    vecmem::device_vector<std::size_t> device_clusters_per_module(
        clusters_per_module_view);
    device_clusters_per_module[globalIndex] = n_clusters;
}


TRACCC_HOST_DEVICE
bool find_clusters_cell_parallel(
    std::size_t globalIndex, const cell_container_types::const_view& cells_view,
    vecmem::data::vector_view<std::size_t> cell_to_module_view,
    vecmem::data::vector_view<std::size_t> cell_indices_in_mod_view,
    vecmem::data::jagged_vector_view<unsigned int> cell_cluster_label_view,
    unsigned int& neighbour_index) {
        
    // Initialize the device container for cells
    cell_container_types::const_device cells_device(cells_view);
    // Do the same with cell to module and cell index mapping
    vecmem::device_vector<std::size_t> device_cell_to_module(
        cell_to_module_view);
    vecmem::device_vector<std::size_t> device_cell_indices_in_mod(
        cell_indices_in_mod_view);
    
    // Ignore if idx is out of range
    if (globalIndex >= device_cell_to_module.size())
        return false;

    // get the current module number from the current cell idx
    std::size_t module_number = device_cell_to_module.at(globalIndex);
    std::size_t cell_index = device_cell_indices_in_mod.at(globalIndex);
    if (module_number == 3000) {
        printf("Module 3000: cell_indices_in_module[%d] = %d\n", (int) globalIndex, (int) cell_index);
    }

    // Initialise the jagged device vector for cell cluster indices
    vecmem::jagged_device_vector<unsigned int> device_cell_cluster_labels(
        cell_cluster_label_view);

    // Get the cells for the current module and the cell this thread
    // is looking at
    const auto& cells = cells_device.at(module_number).items;
    // Get the relevant labels, so the ones for this current module
    auto cluster_labels = device_cell_cluster_labels[module_number];

    // run H-K on the individual cell
    bool label_changed = detail::hoshen_kopelman(module_number, cell_index, cells,
                                                 cluster_labels, neighbour_index);
    return label_changed;
}

TRACCC_HOST_DEVICE
void normalise_cluster_numbers(
    std::size_t module_number,
    vecmem::data::jagged_vector_view<unsigned int> cell_cluster_label_view,
    vecmem::data::vector_view<std::size_t> clusters_per_module_view) {

    // first get the device vectors from the views
    vecmem::jagged_device_vector<unsigned int> device_cell_cluster_labels(
        cell_cluster_label_view);
    vecmem::device_vector<std::size_t> device_clusters_per_module(
        clusters_per_module_view);
    
    auto cluster_labels = device_cell_cluster_labels[module_number];

    unsigned int n_clusters = detail::normalise_cluster_numbers(cluster_labels);

    device_clusters_per_module[module_number] = n_clusters;

}

void set_init_cluster_labels(
    unsigned int globalIndex,
    vecmem::data::vector_view<std::size_t> cell_to_module_view,
    vecmem::data::vector_view<std::size_t> cell_indices_in_mod_view,
    vecmem::data::jagged_vector_view<unsigned int> cell_cluster_label_view) {

    // make relevant device vectors
    vecmem::device_vector<std::size_t> device_cell_to_module(
        cell_to_module_view);
    vecmem::device_vector<std::size_t> device_cell_indices_in_mod(
        cell_indices_in_mod_view);
    vecmem::jagged_device_vector<unsigned int> device_cell_cluster_labels(
        cell_cluster_label_view);
    
    // get the module number and which cell in the module
    std::size_t module_number = device_cell_to_module[globalIndex];
    std::size_t cell_index = device_cell_indices_in_mod[globalIndex];

    auto cluster_labels = device_cell_cluster_labels[module_number];

    // set the label to the index + 1, since labels are 1...N, not 0 indexed
    detail::setup_cluster_labels(cell_index, cluster_labels);
}

}  // namespace traccc::device