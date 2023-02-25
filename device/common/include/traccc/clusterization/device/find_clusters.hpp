/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/clusterization/detail/sparse_ccl.hpp"
#include "traccc/clusterization/detail/hoshen_kopelman.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/cell.hpp"

// Vecmem include(s).
#include <vecmem/containers/data/jagged_vector_view.hpp>
#include <vecmem/containers/data/vector_view.hpp>

// System include(s).
#include <cstddef>

namespace traccc::device {

/// Function that sets up the labels for each cell, this is just
/// the cell number in the module, so they all have a unique label
/// to begin with when clusterisation begins later.
TRACCC_HOST_DEVICE
void set_init_cluster_labels(
    unsigned int globalIndex,
    vecmem::data::vector_view<std::size_t> cell_to_module_view,
    vecmem::data::vector_view<std::size_t> cell_indices_in_mod_view,
    vecmem::data::jagged_vector_view<unsigned int> cell_cluster_label_view);

/// Function that finds the clusters using sparse_ccl algorithm
///
/// It saves the cluster indices for each module in a jagged vector
/// and it counts how many clusters in total were found
///
/// @param[in] globalIndex                  The index of the current thread
/// @param[in] cells_view                   The cells for each module
/// @param[out] sparse_ccl_indices_view     Jagged vector that maps cells to
/// corresponding clusters
/// @param[out] clusters_per_module_view    Vector of numbers of clusters found
/// in each module
///
TRACCC_HOST_DEVICE
void find_clusters(
    std::size_t globalIndex, const cell_container_types::const_view& cells_view,
    vecmem::data::jagged_vector_view<unsigned int> sparse_ccl_indices_view,
    vecmem::data::vector_view<std::size_t> clusters_per_module_view);

// overload the function with the cell parallelised one
TRACCC_HOST_DEVICE
void find_clusters_cell_parallel(
    std::size_t globalIndex, const cell_container_types::const_view& cells_view,
    vecmem::data::vector_view<std::size_t> cell_module_view,
    vecmem::data::vector_view<std::size_t> cell_indices_in_mod_view,
    vecmem::data::jagged_vector_view<unsigned int> cell_cluster_label_view,
    vecmem::data::vector_view<std::size_t> clusters_per_module_view);

// Same as the above, but doing the lookups before in .cu file
TRACCC_HOST_DEVICE
bool find_clusters_cell_parallel_passthrough(
    std::size_t module_number, unsigned int cell_index,
    const vecmem::device_vector<const traccc::cell>& cells,
    vecmem::device_vector<unsigned int>& labels,
    unsigned int& neighbour_index);

// find a nearest neighbour above/left and write its label into
// the current label.
TRACCC_HOST_DEVICE
void write_from_NN(unsigned int cell_index,
                   const vecmem::device_vector<const traccc::cell>& cells,
                   vecmem::device_vector<unsigned int>& labels);

// find the origin of the cluster for each cell, implementation
// of find() part of a union-find clusterisation algorithm
TRACCC_HOST_DEVICE
void hk_find(
    unsigned int cell_index, 
    vecmem::device_vector<unsigned int>& labels);

// function for setting cluster numbers to 1->N only, N distinct clusters
TRACCC_HOST_DEVICE
void normalise_cluster_numbers(
    std::size_t module_number,
    vecmem::data::jagged_vector_view<unsigned int> cell_cluster_label_view,
    vecmem::data::vector_view<std::size_t> clusters_per_module_view);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/clusterization/device/impl/find_clusters.ipp"