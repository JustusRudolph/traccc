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

/// Helper method to find if cell a is left diagonally above b
///
/// @param a the first cell
/// @param b the second cell
///
/// @return boolean to indicate if first cell is above cell b
TRACCC_HOST_DEVICE inline bool is_diagonal_above_left(traccc::cell a, traccc::cell b) {
    return ((a.channel1 - b.channel1) == 1) && ((b.channel0 - a.channel0) == 1);
}

/// Helper method to find if cell a is left diagonally above b
///
/// @param a the first cell
/// @param b the second cell
///
/// @return boolean to indicate if first cell is above cell b
TRACCC_HOST_DEVICE inline bool is_diagonal_above_right(traccc::cell a, traccc::cell b) {
    return ((a.channel1 - b.channel1) == 1) && ((a.channel0 - b.channel0) == 1);
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
    unsigned int diagonal_left_label = 0;
    unsigned int diagonal_right_label = 0;
    unsigned int neighbour_label = 0;
    
    if ((neighbour_index+1) == 0) {  // -1 means unset
        // check all cells in the module for neighbours of current cell
        for (unsigned int i = 0; i < n_cells; i++) {
            // if one neighbour has been found, break out
            //if (left_label * above_label > 0) { break;}  // TODO: Try removing this line
            
            if (!(left_label + diagonal_left_label) && is_above(cells[i], cell)) {
                // overwrite and take only above if there is a cell above
                above_label = labels[i];
                neighbour_label = above_label;
                neighbour_index = i+1;  // set for next run, i+1 so it matches
                                        // the other cases
                break;  // everything links to above so can break out
            }
            else if (!(left_label + diagonal_left_label) && is_left(cells[i], cell)) {
                // only get left cell information if top left isn't a neighbour
                left_label = labels[i];
                neighbour_label = neighbour_label * n_cells + left_label;
                neighbour_index = (neighbour_index + 1) * (n_cells + 1) + i+1;
                // above right and left are the unlinked ones
            }
            else if (is_diagonal_right(cells[i], cell)) {
                // left or diag left and diagonal right create problems, need to
                // have them together, neighbour_label right now contains
                // either left_label or diagonal_left_label
                diagonal_right_label = labels[i];
                neighbour_label = neighbour_label * n_cells + diagonal_right_label;
                neighbour_index = (neighbour_index + 1) * (n_cells + 1) + i+1;
            }
            else if (!(left_label + diagonal_left_label) &&
                     is_diagonal_left(cells[i], cell)) {
                // left links to diagonal left so only set diagonal left label
                // if left unset
                diagonal_left_label = labels[i];
                neighbour_label = neighbour_label * n_cells + diagonal_left_label;
                neighbour_index = (neighbour_index + 1) * (n_cells + 1) + i+1;
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
            labels[cell_index] = left_label;

        }
        else if (above_label > 0) {
            // set all with current label to above
            if (module_number == module_to_check) {
                printf("For Cell %d with label %d in Module %d: Neighbour cell above with label %d. Overwrite all with current label.\n",
                    (int) cell_index, (int) label, (int) module_number, (int) above_label);
            }
            labels[cell_index] = above_label;
            // for (unsigned int j = 0; j < n_cells; j++) {
            //     // overwrite all cells with current label to the one above
            //     if (labels[j] == label) {
            //         labels[j] = above_label;
            //     }
            // }
        }
        else if (diagonal_right_label > 0 && include_diagonals) {
            // in case nothing next to, check the diagonals
            if (module_number == module_to_check) {
                printf("For Cell %d with label %d in Module %d: Neighbour cell diagonally right with label %d. Overwrite all with current label.\n",
                    (int) cell_index, (int) label, (int) module_number, (int) diagonal_right_label);
            }
            labels[cell_index] = diagonal_right_label;
            // for (unsigned int j = 0; j < n_cells; j++) {
            //     // overwrite all cells with current label to the one diagonally above
            //     if (labels[j] == label) {
            //         labels[j] = neighbour_label;
            //     }
            // }
        }
        else if (diagonal_left_label > 0 && include_diagonals) {
            // in case nothing next to, check the diagonals
            if (module_number == module_to_check) {
                printf("For Cell %d with label %d in Module %d: Neighbour cell diagonally left with label %d. Overwrite all with current label.\n",
                    (int) cell_index, (int) label, (int) module_number, (int) diagonal_left_label);
            }
            labels[cell_index] = diagonal_left_label;
            // for (unsigned int j = 0; j < n_cells; j++) {
            //     // overwrite all cells with current label to the one diagonally above
            //     if (labels[j] == label) {
            //         labels[j] = neighbour_label;
            //     }
            // }
        }
    }

    else {  // at least second time going through, know where the neighbour(s) are
        unsigned int nn0_index = neighbour_index % (n_cells+1) - 1;
        unsigned int nn1_index = neighbour_index / (n_cells+1) - 2;

        if ((nn1_index + 2) == 0) {
            // only one neighbour
            neighbour_label = labels[nn0_index]

        neighbour_label = labels[nn0_index];
        // TODO how to merge the two? Just take nn0 for now
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
