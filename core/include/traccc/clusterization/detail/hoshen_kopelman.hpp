/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/cell.hpp"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>

// System include(s).
#include <cassert>

namespace traccc {


namespace detail {

/// Find function as part of union-find algo
/// Finds which cluster the current cell belongs to by looping through
/// until getting to root cell
///
/// @param cell_index is the index of the current cell in the cell collection
/// @param labels is the vector of the output indices (to which cluster a cell
///               belongs to)
/// 
TRACCC_HOST_DEVICE
inline void hk_find(unsigned int cell_index,
                    vecmem::device_vector<unsigned int>& labels) {
    
    unsigned int curr_index = cell_index;
    unsigned int n_cells = labels.size();

    // labels are initially just n_cells + index + 1
    // so if label is not larger than n_cells, have hit the cluster origin
    while (labels[curr_index] > n_cells) {
        curr_index = labels[curr_index] - n_cells - 1;  // to check next point
    }
    // we reach the point of the origin
    labels[cell_index] = labels[curr_index];
}

/// Set up the initial cluster labels and find the nearest neighbour index
///
/// @param cell_index is the index of the current cell in the cell collection
/// @param cells is the cell collection. Sorted by column
/// @param labels is the vector which contains which cluster a cell
///               belongs to
template <typename cell_container_t, typename label_vector>
TRACCC_HOST_DEVICE
unsigned int setup_cluster_labels_and_NN(
    std::size_t cell_index, const cell_container_t& cells,
    label_vector& labels) {
    
    unsigned int n_cells = cells.size();
    traccc::cell cell = cells[cell_index];

    // will be overwritten as less if we find a neighbour
    unsigned int NN_index = n_cells;
 
    // check all cells in the module from current cell to the start
    for (unsigned int i = 1; i <= cell_index; i++) {
        // if one neighbour has been found, break out
        // move steps up from the current cell
        unsigned int index_to_check = cell_index - i;
        traccc::cell cell_to_check = cells[index_to_check];

        // since it's sorted, this is necessarily positive
        unsigned int row_diff = cell.channel1 - cell_to_check.channel1;
        
        if (row_diff > 1) { break; } // if outside row range, there will be no more found

        // within reasonable territory
        int col_diff = cell.channel0 - cell_to_check.channel0;
        if (col_diff * col_diff <= 1) {  // within one cell
            // have found a neighbour that is above/left, write label
            // to point to that neighbour
            NN_index = index_to_check;
            labels[cell_index] = n_cells + index_to_check + 1;
        }
    }
    return NN_index;
}

}  // namespace detail

}  // namespace traccc
