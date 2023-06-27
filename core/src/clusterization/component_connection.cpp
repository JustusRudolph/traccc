/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/clusterization/component_connection.hpp"

#include "traccc/clusterization/detail/sparse_ccl.hpp"
#include "traccc/clusterization/detail/fconn.hpp"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/vector.hpp>

namespace traccc {

component_connection::output_type component_connection::operator()(
    const cell_container_types::host& cells) const {

    std::vector<std::size_t> num_clusters(cells.size(), 0);
    std::vector<std::vector<unsigned int>> clust_indices(cells.size());

    for (std::size_t i = 0; i < cells.size(); i++) {
        const auto& cells_per_module = cells.get_items()[i];

        // Run SparseCCL to fill CCL indices
        //num_clusters[i] = detail::sparse_ccl(cells_per_module, clust_indices[i]);

        // FCONN:
        unsigned int n_cells_per_module = cells_per_module.size();
        clust_indices[i] = std::vector<unsigned int>(n_cells_per_module);

        //Run FCONN to fill clusterisation indices
        for (std::size_t j = 0; j < n_cells_per_module; j++) {
            // FCONN for each cell
            unsigned int nn_index =
                detail::setup_cluster_labels_and_NN(j, cells_per_module, clust_indices[i]);
            
            if (nn_index == n_cells_per_module) {
                // increment number of clusters in module then write to the
                // found cluster origin
                clust_indices[i][j] = ++num_clusters[i];
            }
        }
        for (std::size_t j = 0; j < n_cells_per_module; j++) {
            // trace back for each cell
            detail::fconn_find(j, clust_indices[i]);
        }
    
    }

    // Get total number of clusters
    const std::size_t N =
        std::accumulate(num_clusters.begin(), num_clusters.end(), 0);

    // Create the result container.
    output_type result(N, &(m_mr.get()));

    std::size_t stack = 0;
    for (std::size_t i = 0; i < cells.size(); i++) {

        auto& cells_per_module = cells.get_items()[i];

        // Fill the module link
        std::fill(result.get_headers().begin() + stack,
                  result.get_headers().begin() + stack + num_clusters[i], i);

        // Full the cluster cells
        for (std::size_t j = 0; j < clust_indices[i].size(); j++) {

            auto cindex = static_cast<unsigned int>(clust_indices[i][j] - 1);

            result.get_items()[stack + cindex].push_back(cells_per_module[j]);
        }

        stack += num_clusters[i];
    }

    return result;
}

}  // namespace traccc
