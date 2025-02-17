/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/efficiency/finding_performance_writer.hpp"
#include "traccc/finding/finding_algorithm.hpp"
#include "traccc/fitting/fitting_algorithm.hpp"
#include "traccc/fitting/kalman_filter/kalman_fitter.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/read_measurements.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/options/common_options.hpp"
#include "traccc/options/finding_input_options.hpp"
#include "traccc/options/handle_argument_errors.hpp"
#include "traccc/options/propagation_options.hpp"
#include "traccc/resolution/fitting_performance_writer.hpp"
#include "traccc/utils/seed_generator.hpp"

// Detray include(s).
#include "detray/core/detector.hpp"
#include "detray/detectors/toy_metadata.hpp"
#include "detray/io/common/detector_reader.hpp"
#include "detray/propagator/navigator.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/propagator/rk_stepper.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// System include(s).
#include <exception>
#include <iomanip>
#include <iostream>

using namespace traccc;
namespace po = boost::program_options;

int seq_run(const traccc::finding_input_config& i_cfg,
            const traccc::propagation_options<scalar>& propagation_opts,
            const traccc::common_options& common_opts) {

    /// Type declarations
    using host_detector_type =
        detray::detector<detray::toy_metadata<>, covfie::field,
                         detray::host_container_types>;

    using b_field_t = typename host_detector_type::bfield_type;
    using rk_stepper_type =
        detray::rk_stepper<b_field_t::view_t, traccc::transform3,
                           detray::constrained_step<>>;

    using host_navigator_type = detray::navigator<const host_detector_type>;
    using host_fitter_type =
        traccc::kalman_fitter<rk_stepper_type, host_navigator_type>;

    // Memory resources used by the application.
    vecmem::host_memory_resource host_mr;

    // Performance writer
    traccc::finding_performance_writer find_performance_writer(
        traccc::finding_performance_writer::config{});
    traccc::fitting_performance_writer fit_performance_writer(
        traccc::fitting_performance_writer::config{});

    /*****************************
     * Build a geometry
     *****************************/

    // B field value and its type
    // @TODO: Set B field as argument
    const traccc::vector3 B{0, 0, 2 * detray::unit<traccc::scalar>::T};

    // Read the detector
    detray::io::detector_reader_config reader_cfg{};
    reader_cfg
        .add_file(traccc::io::data_directory() + common_opts.detector_file)
        .add_file(traccc::io::data_directory() + common_opts.material_file)
        .bfield_vec(B[0], B[1], B[2]);

    const auto [host_det, names] =
        detray::io::read_detector<host_detector_type>(host_mr, reader_cfg);

    const auto surface_transforms = traccc::io::alt_read_geometry(host_det);

    /*****************************
     * Do the reconstruction
     *****************************/

    // Standard deviations for seed track parameters
    static constexpr std::array<traccc::scalar, traccc::e_bound_size> stddevs =
        {1e-4 * detray::unit<traccc::scalar>::mm,
         1e-4 * detray::unit<traccc::scalar>::mm,
         1e-3,
         1e-3,
         1e-4 / detray::unit<traccc::scalar>::GeV,
         1e-4 * detray::unit<traccc::scalar>::ns};

    // Finding algorithm configuration
    typename traccc::finding_algorithm<rk_stepper_type,
                                       host_navigator_type>::config_type cfg;
    cfg.min_track_candidates_per_track = i_cfg.track_candidates_range[0];
    cfg.max_track_candidates_per_track = i_cfg.track_candidates_range[1];
    cfg.constrained_step_size = propagation_opts.step_constraint;

    // few tracks (~1 out of 1000 tracks) are missed when chi2_max = 15
    cfg.chi2_max = 30.f;

    // Finding algorithm object
    traccc::finding_algorithm<rk_stepper_type, host_navigator_type>
        host_finding(cfg);

    // Fitting algorithm object
    typename traccc::fitting_algorithm<host_fitter_type>::config_type fit_cfg;
    fit_cfg.step_constraint = propagation_opts.step_constraint;
    traccc::fitting_algorithm<host_fitter_type> host_fitting(fit_cfg);

    // Seed generator
    traccc::seed_generator<host_detector_type> sg(host_det, stddevs);

    // Iterate over events
    for (unsigned int event = common_opts.skip;
         event < common_opts.events + common_opts.skip; ++event) {

        // Truth Track Candidates
        traccc::event_map2 evt_map2(event, common_opts.input_directory,
                                    common_opts.input_directory,
                                    common_opts.input_directory);

        traccc::track_candidate_container_types::host truth_track_candidates =
            evt_map2.generate_truth_candidates(sg, host_mr);

        // Prepare truth seeds
        traccc::bound_track_parameters_collection_types::host seeds(&host_mr);
        const unsigned int n_tracks = truth_track_candidates.size();
        for (unsigned int i_trk = 0; i_trk < n_tracks; i_trk++) {
            seeds.push_back(truth_track_candidates.at(i_trk).header);
        }

        // Read measurements
        traccc::measurement_container_types::host measurements_per_event =
            traccc::io::read_measurements_container(
                event, common_opts.input_directory, traccc::data_format::csv,
                &host_mr);

        // Run finding
        auto track_candidates =
            host_finding(host_det, measurements_per_event, seeds);

        std::cout << "Number of found tracks: " << track_candidates.size()
                  << std::endl;

        // Run fitting
        auto track_states = host_fitting(host_det, track_candidates);

        std::cout << "Number of fitted tracks: " << track_states.size()
                  << std::endl;

        const unsigned int n_fitted_tracks = track_states.size();

        if (common_opts.check_performance) {
            find_performance_writer.write(traccc::get_data(track_candidates),
                                          evt_map2);

            for (unsigned int i = 0; i < n_fitted_tracks; i++) {
                const auto& trk_states_per_track = track_states.at(i).items;

                const auto& fit_info = track_states[i].header;

                fit_performance_writer.write(trk_states_per_track, fit_info,
                                             host_det, evt_map2);
            }
        }
    }

    if (common_opts.check_performance) {
        find_performance_writer.finalize();
        fit_performance_writer.finalize();
    }

    return 1;
}

// The main routine
//
int main(int argc, char* argv[]) {
    // Set up the program options
    po::options_description desc("Allowed options");

    // Add options
    desc.add_options()("help,h", "Give some help with the program's options");
    traccc::common_options common_opts(desc);
    traccc::finding_input_config finding_input_cfg(desc);
    traccc::propagation_options<scalar> propagation_opts(desc);

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    // Check errors
    traccc::handle_argument_errors(vm, desc);

    // Read options
    common_opts.read(vm);
    finding_input_cfg.read(vm);
    propagation_opts.read(vm);

    std::cout << "Running " << argv[0] << " " << common_opts.input_directory
              << " " << common_opts.events << std::endl;

    return seq_run(finding_input_cfg, propagation_opts, common_opts);
}
