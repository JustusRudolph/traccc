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
    return ((a.channel1 - b.channel1) == 1) && (b.channel0 == a.channel0);
}

/// Helper method to find if cell a is diagonally above b
///
/// @param a the first cell
/// @param b the second cell
///
/// @return boolean to indicate if first cell is above cell b
TRACCC_HOST_DEVICE inline bool is_diagonal_above(traccc::cell a, traccc::cell b) {
    return ((a.channel1 - b.channel1) == 1) &&
            ((b.channel0 - a.channel0)*(b.channel0 - a.channel0) == 1);
}

/// Find function as part of union-find algo
/// Finds which cluster the current cell belongs to by looping through
/// until getting to root cell
///
/// @param cell_index is the index of the current cell in the cell collection
/// @param cells is the cell collection
/// @param labels is the vector of the output indices (to which cluster a cell
///               belongs to)
/// @param neighbour_index is the index of the neighbour to overwrite from
/// 
TRACCC_HOST_DEVICE
inline void hk_find(unsigned int cell_index,
                    vecmem::device_vector<unsigned int>& labels) {
    
    unsigned int curr_index = cell_index;

    // labels are initially just the index + 1
    // so if the current label is the same as the current index+1, then
    // we are at the origin of the cluster
    while (curr_index != (labels[curr_index] - 1)) {
        curr_index = labels[curr_index] - 1;  // to check next point
    }
    // we reach the point of the origin
    labels[cell_index] = labels[curr_index];
}

/// Hoshen-Kopelman Clusterization algorithm
///
/// @param globalIndex is the current module, used for debugging only
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

/// Hoshen-Kopelman Clusterization algorithm overloaded for single cell
/// parallelisation
///
/// @param module_number is the current module, only used for debug purposes
/// @param cell_index is the index of the current cell in the cell collection
/// @param cells is the cell collection
/// @param labels is the vector of the output indices (to which cluster a cell
///               belongs to)
/// @param neighbour_index is the index of the neighbour to overwrite from
/// @return whether the current label changed in this iteration of HK
template <typename cell_container_t, typename label_vector>
TRACCC_HOST_DEVICE
bool hoshen_kopelman(std::size_t module_number, unsigned int cell_index,
                    const cell_container_t& cells, label_vector& labels,
                    unsigned int& neighbour_index) {
    
    unsigned int module_to_check = -1;  // pick which to probe with debug
    bool include_diagonals = true;  // for debugging

    unsigned int n_cells = cells.size();
    traccc::cell cell = cells[cell_index];
    unsigned int label = labels[cell_index];

    // initialise labels to non-label integer (i.e. <1)
    // first three are for debug purposes only
    unsigned int left_label = 0;
    unsigned int above_label = 0;
    unsigned int diagonal_above_label = 0;
    unsigned int neighbour_label = 0;
    
    if (neighbour_index > n_cells) {
        // check all cells in the module for neighbours of current cell
        for (unsigned int i = 0; i < n_cells; i++) {
            // if one neighbour has been found, break out
            //if (left_label * above_label > 0) { break;}  // TODO: Try removing this line

            if (is_above(cells[i], cell)) {
                neighbour_label = labels[i];
                neighbour_index = i;  // set for next run
                above_label = neighbour_label;  // debug
                break;
            }
            else if (is_left(cells[i], cell)) {
                neighbour_label = labels[i];
                neighbour_index = i;
                left_label = neighbour_label;  // debug
                break;
            }
            // following assumes no double diagonal without above set too
            else if (is_diagonal_above(cells[i], cell)) {
                neighbour_label = labels[i];
                neighbour_index = i;
                diagonal_above_label = neighbour_label;  // debug
                break;
            }
        }
        if (left_label > 0) {
            // set all with current label to left
            if (module_number == module_to_check) {
                printf("For Cell %d with label %d in Module %d: Neighbour cell left with label %d. Overwrite all with current label.\n",
                    (int) cell_index, (int) label, (int) module_number, (int) left_label);
            }
            // for (unsigned int j = 0; j < n_cells; j++) {
            //     // overwrite all cells with current label to the one left
            //     if (labels[j] == label) {
            //         labels[j] = left_label;
            //     }
            // }
            labels[cell_index] = neighbour_label;

        }
        else if (above_label > 0) {
            // set all with current label to above
            if (module_number == module_to_check) {
                printf("For Cell %d with label %d in Module %d: Neighbour cell above with label %d. Overwrite all with current label.\n",
                    (int) cell_index, (int) label, (int) module_number, (int) above_label);
            }
            labels[cell_index] = neighbour_label;
            // for (unsigned int j = 0; j < n_cells; j++) {
            //     // overwrite all cells with current label to the one above
            //     if (labels[j] == label) {
            //         labels[j] = above_label;
            //     }
            // }
        }
        else if (diagonal_above_label > 0 && include_diagonals) {
            // in case nothing next to, check the diagonals
            if (module_number == module_to_check) {
                printf("For Cell %d with label %d in Module %d: Neighbour cell diagonally above with label %d. Overwrite all with current label.\n",
                    (int) cell_index, (int) label, (int) module_number, (int) diagonal_above_label);
            }
            for (unsigned int j = 0; j < n_cells; j++) {
                // overwrite all cells with current label to the one diagonally above
                if (labels[j] == label) {
                    labels[j] = neighbour_label;
                }
            }
        }
    }

    else {  // at least second time going through, know where the neighbour is
        neighbour_label = labels[neighbour_index];
        labels[cell_index] = neighbour_label;
    }
    // now decision tree for what to do with neighbour information
    // if (left_label * above_label > 0) {
    //     // make union
    //     if (module_number == module_to_check) {
    //         printf("For Cell %d with label %d in Module %d: Neighbour cells left and above with labels %d and %d. Overwrite all with current and left label.\n",
    //             (int) cell_index, (int) label, (int) module_number, (int) left_label, (int) above_label);
    //     }
    //     for (unsigned int j = 0; j < n_cells; j++) {
    //         // overwrite all labels left and the connected labels of the current cell
    //         if (labels[j] == left_label || labels[j] == label) {
    //             labels[j] = above_label;
    //         }
    //     }
    // }
    
    // return whether current label_changed
    bool label_changed = labels[cell_index] != label;
    return label_changed;
}

/// Find the nearest neighbour of the current cell being probed. In order, it
/// will select left, diag left, above, diag right as its sole nearest neighbour.
/// Then, write the cluster label from that nearest neighbour into current.
///
/// @param cell_index is the index of the current cell in the cell collection
/// @param cells is the cell collection. Sorted by column
/// @param labels is the vector of the output indices (to which cluster a cell
///               belongs to)
template <typename cell_container_t, typename label_vector>
TRACCC_HOST_DEVICE
void write_from_NN(unsigned int cell_index, const cell_container_t& cells,
                   label_vector& labels) {
    
    unsigned int n_cells = cells.size();
    traccc::cell cell = cells[cell_index];
 
    // check all cells in the module from current cell to the start
    for (unsigned int i = 1; i <= cell_index; i++) {
        // if one neighbour has been found, break out
        // move steps up from the current cell
        unsigned int index_to_check = cell_index - i;
        traccc::cell cell_to_check = cells[index_to_check];

        // since it's sorted, this is necessarily positive
        unsigned int row_diff = cell.channel1 - cell_to_check.channel1;
        
        if (row_diff > 1) { break;}  // out of bounds

        unsigned int col_diff = cell.channel0 - cell_to_check.channel0;
        if (col_diff * col_diff <= 1) {  // within one cell
            // have found a neighbour that is above/left, write its label
            labels[cell_index] = labels[index_to_check];
            //break;
        }
    }
}

/// Simple helper function to set the initial cluster labels
///
/// @param cell_index is the index of the current cell in the cell collection
/// @param labels is the vector which contains which cluster a cell
///               belongs to
TRACCC_HOST_DEVICE
void setup_cluster_labels(std::size_t cell_index,
                          vecmem::device_vector<unsigned int>& labels) {
    
    labels[cell_index] = cell_index + 1;
}

/// Normalisation of cluster numbers. This sets the cluster labels s.t. all
/// labels are smaller than the total number of clusters. I.e., if there are
/// N clusters, the labels will just be in range of 1->N. This function is
/// parallelised by module.
///
/// @param labels is the vector containing the cluster labels of each cell
/// @return The number of clusters in the module.
template<typename label_vector>
TRACCC_HOST_DEVICE
inline unsigned int normalise_cluster_numbers(label_vector& labels) {
    
    unsigned int num_clusters = 0;
    unsigned int n_cells = labels.size();
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

}  // namespace detail

}  // namespace traccc
