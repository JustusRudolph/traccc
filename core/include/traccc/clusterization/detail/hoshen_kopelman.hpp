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

/// Implemementation of Hoshen-Kopelman, following
/// https://en.wikipedia.org/wiki/Hoshen%E2%80%93Kopelman_algorithm
///
namespace detail {


/// Create a union of two entries @param e1 and @param e2
///
/// @param L an equivalance table
///
/// @return the rleast common ancestor of the entries
template <typename ccl_vector_t>
TRACCC_HOST_DEVICE inline unsigned int make_union2(ccl_vector_t& L,
                                                  unsigned int e1,
                                                  unsigned int e2) {

    int e;
    if (e1 < e2) {
        e = e1;
        assert(e2 < L.size());
        L[e2] = e;
    } else {
        e = e2;
        assert(e1 < L.size());
        L[e1] = e;
    }
    return e;
}

/// Helper method to find adjacent cells
///
/// @param a the first cell
/// @param b the second cell
///
/// @return boolan to indicate 8-cell connectivity
TRACCC_HOST_DEVICE inline bool is_adjacent2(traccc::cell a, traccc::cell b) {
    return (a.channel0 - b.channel0) * (a.channel0 - b.channel0) <= 1 and
           (a.channel1 - b.channel1) * (a.channel1 - b.channel1) <= 1;
}

/// Helper method to find if cell a is left of cell b
///
/// @param a the first cell
/// @param b the second cell
///
/// @return boolean to indicate if first cell is left of cell b
TRACCC_HOST_DEVICE inline bool is_left(traccc::cell a, traccc::cell b) { 
    // if (globalIdx == 2464) {
    //     printf("check_cell: (x,y) = (%d, %d); curr_cell: (x,y) = (%d, %d)",
    //            a.channel0, a.channel1, b.channel0, b.channel1);
    //     printf("I.e. curr_cell - check_cell = (%d, %d)\n",
    //            b.channel0 - a.channel0, b.channel1 - a.channel1);
    // }
    return ((b.channel0 - a.channel0) == 1) && (b.channel1 == a.channel1);
}

/// Helper method to find if cell a is above cell b
///
/// @param a the first cell
/// @param b the second cell
///
/// @return boolean to indicate if first cell is above cell b
TRACCC_HOST_DEVICE inline bool is_above(traccc::cell a, traccc::cell b) {
    return ((b.channel1 - a.channel1) == 1) && (b.channel0 == a.channel0);
}

/// Helper method to find if cell a is diagonally above b
///
/// @param a the first cell
/// @param b the second cell
///
/// @return boolean to indicate if first cell is above cell b
TRACCC_HOST_DEVICE inline bool is_diagonal_above(traccc::cell a, traccc::cell b) {
    return ((b.channel1 - a.channel1) == 1) &&
            ((b.channel0 - a.channel0)*(b.channel0 - a.channel0) == 1);
}


/// Helper method to find define distance,
/// does not need abs, as channels are sorted in
/// column major
///
/// @param a the first cell
/// @param b the second cell
///
/// @return boolan to indicate !8-cell connectivity
TRACCC_HOST_DEVICE inline bool is_far_enough2(traccc::cell a, traccc::cell b) {
    return (a.channel1 - b.channel1) > 1;
}

/// Hoshen-Kopelman Clusterization algorithm
///
/// @param cells is the cell collection
/// @param labels is the vector of the output indices (to which cluster a cell
/// belongs to)
/// @return number of clusters
template <typename cell_container_t, typename label_vector>
TRACCC_HOST_DEVICE inline unsigned int hoshen_kopelman(std::size_t globalIndex, 
                                                       const cell_container_t& cells,
                                                       label_vector& labels) {
    // first specify if we want to print debug messages throughout
    bool print_debug = false; //globalIndex == 2409;   
    bool include_diagonals = true;  // change this to get graph for output
                                    // where HK only takes nns into account                                                  
    // The number of cells.
    const unsigned int n_cells = cells.size();

    // give each cell a unique label to begin with
    for (unsigned int i = 0; i < n_cells; i++) {
        labels[i] = i+1;
    }
    // naive search for left and above O(n)
    for (unsigned int i = 0; i < n_cells; i++) {
        traccc::cell curr_cell = cells[i];
        unsigned int left_label = 0;
        unsigned int above_label = 0;
        unsigned int diagonal_above_label = 0;
        if (print_debug) {
            printf("Cell number %d with label %d: Position (%d, %d).\n",
                   i, i+1, curr_cell.channel0, curr_cell.channel1);
        }
        
        // only check the cells that haven't been checked yet
        for (unsigned int j = 0; j < n_cells; j++) {
            // if one above and on left has been found, break out
            if (left_label * above_label > 0) { break;}

            if (is_left(cells[j], curr_cell)) {
                left_label = labels[j];
                continue;
            }
            else if (is_above(cells[j], curr_cell)) {
                above_label = labels[j];
                continue;
            }
            // following assumes no double diagonal without above set too
            else if (is_diagonal_above(cells[j], curr_cell)) {
                diagonal_above_label = labels[j];
            }
        }
        if (print_debug) {
            printf("Above label: %d, left label: %d\n", above_label, left_label);
        }
        // now choose what to do with current cell label
        if (left_label * above_label > 0) {
            // make union
            for (unsigned int j = 0; j < n_cells; j++) {
                // overwrite all labels left and the connected labels to curr cell
                if (labels[j] == left_label || labels[j] == labels[i]) {
                    labels[j] = above_label;
                }
            }
        }
        else if (left_label > 0) {
            // set all with current label to left
            for (unsigned int j = 0; j < n_cells; j++) {
                // overwrite all cells with current label to the one left
                if (labels[j] == labels[i]) {
                    labels[j] = left_label;
                }
            }
        }
        else if (above_label > 0) {
            // set all with current label to above
            for (unsigned int j = 0; j < n_cells; j++) {
                // overwrite all cells with current label to the one above
                if (labels[j] == labels[i]) {
                    labels[j] = above_label;
                }
            }
        }
        else if (diagonal_above_label > 0 && include_diagonals) {
            // in case nothing next to, check the diagonals
            for (unsigned int j = 0; j < n_cells; j++) {
                // overwrite all cells with current label to the one diagonally above
                if (labels[j] == labels[i]) {
                    labels[j] = diagonal_above_label;
                }
            }
        }
        if (print_debug) {
            printf("Following H-K: labels of the cells are: [");
            for (unsigned int j = 0; j < n_cells; j++) {
                if (j == 0) {
                    printf("%d", labels[j]);
                }
                else {
                    printf(", %d", labels[j]);
                }
            }
            printf("]\n\n");
        }
    }

    unsigned int num_clusters = 0;
    for (unsigned int i = 0; i < n_cells; i++) {
        unsigned int j = 0;
        for (j = 0; j < i; j++) {
            if (labels[i] == labels[j]) {
                // have already had this label
                break;
            }
        }
        if (i == j) {
            // went through all previous ones and no match, so this one is unique
            num_clusters++;
        }
    }
    unsigned int curr_cluster_number = 1;  // Which one to overwrite with
    for (unsigned int i = 0; i < n_cells; i++) {
        if (labels[i] > num_clusters) {
            // pick which to overwrite with
            unsigned int j = 0;  // iterator
            // how many times since last increment in curr_cluster_number
            // thus, once this hits n_cells, we have gone one way around without
            // finding the "curr_cluster_number", meaning it's not yet set.
            unsigned int n_iterations = 0;

            while (n_iterations < n_cells) {
                if (labels[j] == curr_cluster_number) {
                    curr_cluster_number++;  // check for the next one
                    n_iterations = 0;
                }
                n_iterations++;
                j++;
                j = j % n_cells;  // loop back to front if end reached
            }
            // we have now picked which to overwrite with
            for (unsigned int k = i+1; k < n_cells; k++) {
                if (labels[k] == labels[i]) {
                    labels[k] = curr_cluster_number;
                }
            }
            // lastly, overwrite the initial one too
            labels[i] = curr_cluster_number;
        }

    }

    return num_clusters;
}
// overload Hoshen-Kopelman with parallelisation of just one cell
template <typename cell_container_t, typename label_vector>
TRACCC_HOST_DEVICE
inline unsigned int hoshen_kopelman(std::size_t globalIndex, 
                                    const cell_container_t& cells,
                                    const traccc::cell cell;
                                    label_vector& labels) {
    return 0
}                                                        

}  // namespace detail

}  // namespace traccc
