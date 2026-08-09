#ifndef MAT_SUBSET_EXPERIMENTS_TESTER_H
#define MAT_SUBSET_EXPERIMENTS_TESTER_H

#include <chrono>     // For timestamps
#include <cmath>      // For std::log, std::exp
#include <ctime>      // For std::ctime
#include <limits>     // For std::numeric_limits
#include <execution>  // For std::execution::par
#include <filesystem> // For std::filesystem::path
#include <format>     // For std::format
#include <fstream>    // For std::ofstream
#include <iostream>   // For reading from and writing to the console
#include <memory>     // For std::unique_ptr
#include <mutex>      // For std::mutex
#include <numeric>    // For std::iota
#include <string>     // For std::string
#include <thread>     // For std::thread::hardware_concurrency
#include <vector>     // For std::vector

#include <Eigen/Core> // For vectors and matrices
#include <Eigen/QR>   // For Eigen::HouseholderQR
#include <Eigen/SVD>  // For Eigen::BDCSVD

#include <nlohmann/json.hpp> // For parsing .json files

#include <MatSubset/MatSubset.h> // For subset selection algorithms and utils

#include "MatrixGeneratorFactory.h" // For matrix generator factory
#include "ProgressBar.h"            // For progress bar
#include "SelectorFactory.h"        // For selector factory
#include "Utils.h"                  // For add_underscores function

namespace MatSubset::Experiments {

template <typename Scalar> class Tester {
  public:
    Tester(const nlohmann::json &experiments_config,
           const std::filesystem::path &output_path)
        : experiments_config(experiments_config), output_path(output_path) {
        // Factories are automatically populated via DefaultSelectorFactory
        // and DefaultMatrixGeneratorFactory constructors
    }

    void runAll() {
        size_t num_experiments = experiments_config.size();
        size_t current_experiment = 0;

        // Track experiment names for index.json
        std::vector<std::string> experiment_names;

        for (auto &experiment_config : experiments_config) {
            current_experiment++;
            std::string experiment_name =
                experiment_config.at("name").get<std::string>();

            std::cout << "\n[" << current_experiment << "/" << num_experiments
                      << "] Running experiment: " << experiment_name
                      << std::endl;

            // Only add to index if experiment actually runs
            bool ran = runExperiment(experiment_config);
            if (ran) {
                experiment_names.push_back(experiment_name);
            }
        }

        // Write index.json to the root output folder
        writeIndexFile(experiment_names);

        std::cout << "\n";
    }

  private:
    const nlohmann::json experiments_config; //< Description of experiments
    const std::filesystem::path output_path; //< Path to the output directory

    // Factory instances for creating selectors and matrix generators
    DefaultSelectorFactory<Scalar> selector_factory;
    DefaultMatrixGeneratorFactory<Scalar> matrix_generator_factory;

    // Function to run single experiment
    bool runExperiment(const nlohmann::json &experiment_config) {
        // Check if the experiment is enabled
        if (!experiment_config.value("enabled", true)) {
            std::cout << "Experiment disabled, skipping..." << std::endl;
            return false;
        }

        // Construct the vector of relevant selectors and their display names
        std::vector<std::unique_ptr<SelectorBase<Scalar>>> selectors;
        std::vector<std::string> selector_names;
        const auto &algorithms_json = experiment_config.at("algorithms");

        for (const auto &algorithm_config : algorithms_json) {
            auto selector_ptr = selector_factory.create(algorithm_config);
            selectors.push_back(std::move(selector_ptr));
            // Use "display_name" if provided, otherwise fall back to "name"
            selector_names.push_back(
                algorithm_config.value("display_name",
                    algorithm_config.at("name").get<std::string>()));
        }

        // Construct the matrix generator
        const auto &matrix_config = experiment_config.at("matrix");
        auto matrix_generator = matrix_generator_factory.create(matrix_config);

        // Determine k values
        std::vector<Eigen::Index> k_values;
        auto append_range = [&](const nlohmann::json &range) {
            Eigen::Index start = range.at("start").get<Eigen::Index>();
            Eigen::Index stop  = range.at("stop").get<Eigen::Index>();
            Eigen::Index step  = range.at("step").get<Eigen::Index>();
            for (Eigen::Index k = start; k <= stop; k += step) {
                k_values.push_back(k);
            }
        };
        if (experiment_config.contains("k_values")) {
            k_values = experiment_config.at("k_values")
                           .get<std::vector<Eigen::Index>>();
        } else if (experiment_config.contains("k_values_range")) {
            append_range(experiment_config.at("k_values_range"));
        } else if (experiment_config.contains("k_values_ranges")) {
            for (const auto &range : experiment_config.at("k_values_ranges")) {
                append_range(range);
            }
        }

        int trials_per_k = experiment_config.at("trials_per_k").get<int>();

        // Create experiment subfolder
        std::string experiment_name =
            experiment_config.at("name").get<std::string>();
        std::filesystem::path experiment_folder =
            output_path / Utils::add_underscores(experiment_name);
        std::filesystem::create_directories(experiment_folder);

        // Save experiment configuration for reproducibility
        nlohmann::json saved_config = experiment_config;

        // Replace range specs with the resolved k_values array
        for (const auto &key : {"k_values_range", "k_values_ranges"}) {
            if (saved_config.contains(key)) {
                saved_config.erase(key);
            }
        }
        saved_config["k_values"] = k_values;

        // Remove runtime-specific fields
        saved_config.erase("enabled");

        // Add metadata with start time
        auto start_time = std::chrono::system_clock::now();
        auto start_timestamp = std::chrono::system_clock::to_time_t(start_time);
        saved_config["metadata"]["start_time"] = std::ctime(&start_timestamp);
        saved_config["metadata"]["num_threads"] =
            std::thread::hardware_concurrency();

        // Write initial config (will be updated with finish time later)
        std::filesystem::path config_file_path =
            experiment_folder / "config.json";

        std::vector<std::ofstream> output_files;

        // Check once whether the generator provides a target vector
        auto target_vector_opt = matrix_generator->getTargetVector();
        bool has_target = target_vector_opt.has_value();

        for (size_t i = 0; i < selectors.size(); ++i) {
            std::filesystem::path file_path =
                experiment_folder /
                (Utils::add_underscores(selector_names[i]) + ".csv");
            output_files.emplace_back(file_path);
            output_files.back()
                << "k,pinv_spectral_norm_ratio,pinv_frobenius_norm_ratio,"
                   "X_S_dag_X_spectral_norm_inv,X_S_dag_X_frobenius_norm_inv,"
                   "wall_time_ms,swap_count,spectral_bound,frobenius_bound,"
                   "volume_ratio,regression_mse"
                << std::endl;
        }

        // Main loop with progress tracking
        std::cout << std::endl; // Add newline so progress bar appears below
        ProgressBar progress_bar(k_values.size() * trials_per_k *
                                 selectors.size());

        // Precompute theoretical bounds once per k (independent of trials).
        // spectral_bounds[ki][i] is the bound for k_values[ki], selector i.
        auto [m, n] = matrix_generator->getMatrixSize();
        std::vector<std::vector<Scalar>> spectral_bounds_by_k(k_values.size());
        std::vector<std::vector<Scalar>> frobenius_bounds_by_k(k_values.size());
        for (size_t ki = 0; ki < k_values.size(); ++ki) {
            Eigen::Index k = k_values[ki];
            spectral_bounds_by_k[ki].resize(selectors.size());
            frobenius_bounds_by_k[ki].resize(selectors.size());
            for (size_t i = 0; i < selectors.size(); ++i) {
                spectral_bounds_by_k[ki][i] = std::sqrt(
                    selectors[i]->template bound<Norm::Spectral>(m, n, k));
                frobenius_bounds_by_k[ki][i] = std::sqrt(
                    selectors[i]->template bound<Norm::Frobenius>(m, n, k));
            }
        }

        // Flatten the (k, trial) grid into a single work list so a single
        // parallel_for saturates all cores even when trials_per_k == 1.
        // Each work item is (k_index, trial_index); the trial index is unused
        // by the body but kept for clarity / potential per-trial seeding.
        std::vector<std::pair<size_t, int>> work_items;
        work_items.reserve(k_values.size() * trials_per_k);
        for (size_t ki = 0; ki < k_values.size(); ++ki) {
            for (int t = 0; t < trials_per_k; ++t) {
                work_items.emplace_back(ki, t);
            }
        }

        std::mutex output_mutex;

        std::for_each(
            std::execution::par, work_items.begin(), work_items.end(),
            [&](const std::pair<size_t, int> &item) {
                    size_t ki = item.first;
                    Eigen::Index k = k_values[ki];
                    const std::vector<Scalar> &spectral_bounds =
                        spectral_bounds_by_k[ki];
                    const std::vector<Scalar> &frobenius_bounds =
                        frobenius_bounds_by_k[ki];

                    // generateMatrix() is thread safe
                    Eigen::MatrixX<Scalar> X =
                        matrix_generator->generateMatrix();

                    Eigen::BDCSVD<Eigen::MatrixX<Scalar>> svd_X(X);
                    const auto &sv_X = svd_X.singularValues();

                    Scalar X_dag_spectral_norm =
                        static_cast<Scalar>(1) / sv_X(sv_X.size() - 1);
                    Scalar X_dag_frobenius_norm =
                        std::sqrt(sv_X.array().inverse().square().sum());

                    // log-volume of X: sum of log singular values
                    Scalar log_vol_X = sv_X.array().log().sum();

                    // Compute Q from LQ decomposition of X (X = LQ)
                    const Eigen::Index m_x = X.rows();
                    const Eigen::Index n_x = X.cols();
                    Eigen::HouseholderQR<Eigen::MatrixX<Scalar>> lq(
                        X.transpose());
                    Eigen::MatrixX<Scalar> Q =
                        (lq.householderQ() *
                         Eigen::MatrixX<Scalar>::Identity(n_x, m_x))
                            .transpose();

                    for (int i = 0; i < static_cast<int>(selectors.size());
                         ++i) {
                        auto t0 = std::chrono::steady_clock::now();
                        Eigen::Index swap_count = -1;
                        std::vector<Eigen::Index> selected_indices =
                            selectors[i]->selectSubset(X, k, &swap_count);
                        auto t1 = std::chrono::steady_clock::now();

                        double wall_time_ms =
                            std::chrono::duration<double, std::milli>(t1 - t0)
                                .count();

                        Eigen::MatrixX<Scalar> X_S =
                            X(Eigen::all, selected_indices);
                        Scalar X_S_dag_spectral_norm =
                            MatSubset::Utils::pinv_norm<Scalar, Norm::Spectral>(
                                X_S);
                        Scalar X_S_dag_frobenius_norm =
                            MatSubset::Utils::pinv_norm<Scalar,
                                                        Norm::Frobenius>(X_S);
                        Scalar pinv_spectral_norm_ratio =
                            X_dag_spectral_norm / X_S_dag_spectral_norm;
                        Scalar pinv_frobenius_norm_ratio =
                            X_dag_frobenius_norm / X_S_dag_frobenius_norm;

                        Eigen::MatrixX<Scalar> Q_S =
                            Q(Eigen::all, selected_indices);
                        Scalar Q_S_dag_spectral_norm =
                            MatSubset::Utils::pinv_norm<Scalar, Norm::Spectral>(
                                Q_S);
                        Scalar Q_S_dag_frobenius_norm =
                            MatSubset::Utils::pinv_norm<Scalar,
                                                        Norm::Frobenius>(Q_S);
                        Scalar X_S_dag_X_spectral_norm_inv =
                            static_cast<Scalar>(1) / Q_S_dag_spectral_norm;
                        Scalar X_S_dag_X_frobenius_norm_inv =
                            static_cast<Scalar>(1) / Q_S_dag_frobenius_norm;

                        // Volume ratio: vol(X_S) / vol(X)
                        // vol = product of singular values; computed in
                        // log-space for numerical stability.
                        Eigen::BDCSVD<Eigen::MatrixX<Scalar>> svd_X_S(X_S);
                        Scalar log_vol_X_S =
                            svd_X_S.singularValues().array().log().sum();
                        Scalar volume_ratio =
                            std::exp(log_vol_X_S - log_vol_X);

                        // Regression MSE: fit β on selected samples, evaluate
                        // on all n samples.
                        //
                        // Model: y ≈ X^T β  (X is m×n, β ∈ ℝ^m, y ∈ ℝ^n)
                        // Fit:   β = (X_S^T)^† y_S
                        //          = X_S (X_S^T X_S)^{-1} y_S
                        //   computed via QR of X_S^T (k×m, full column rank
                        //   since k ≥ m is guaranteed by subset selection).
                        // MSE:   ||X^T β - y||² / n
                        Scalar regression_mse =
                            std::numeric_limits<Scalar>::quiet_NaN();
                        if (has_target) {
                            const Eigen::VectorX<Scalar> &y = *target_vector_opt;
                            Eigen::VectorX<Scalar> y_S =
                                y(selected_indices);
                            // Solve X_S^T β = y_S in the least-squares sense
                            // (minimum-norm solution via column-pivoting QR)
                            Eigen::VectorX<Scalar> beta =
                                X_S.transpose()
                                    .colPivHouseholderQr()
                                    .solve(y_S);
                            Eigen::VectorX<Scalar> residual =
                                X.transpose() * beta - y;
                            regression_mse =
                                residual.squaredNorm() /
                                static_cast<Scalar>(y.size());
                        }

                        {
                            std::lock_guard<std::mutex> lock(output_mutex);
                            // Work items across all k run concurrently, so a
                            // "current k" caption would just bounce around;
                            // show the plain bar instead.
                            progress_bar.update();
                            output_files[i]
                                << k << "," << pinv_spectral_norm_ratio << ","
                                << pinv_frobenius_norm_ratio << ","
                                << X_S_dag_X_spectral_norm_inv << ","
                                << X_S_dag_X_frobenius_norm_inv << ","
                                << wall_time_ms << ","
                                << swap_count << ","
                                << spectral_bounds[i] << ","
                                << frobenius_bounds[i] << ","
                                << volume_ratio << ","
                                << regression_mse << std::endl;
                        }
                    }
                });

        // Add finish time and write final config
        auto finish_time = std::chrono::system_clock::now();
        auto finish_timestamp =
            std::chrono::system_clock::to_time_t(finish_time);
        saved_config["metadata"]["finish_time"] = std::ctime(&finish_timestamp);

        std::ofstream config_file(config_file_path);
        config_file << saved_config.dump(4) << std::endl;
        config_file.close();

        std::cout << std::endl;
        return true;
    }

    // Function to write index.json file in the output folder
    void writeIndexFile(const std::vector<std::string> &experiment_names) {
        nlohmann::json index_data;
        index_data["experiments"] = experiment_names;

        std::filesystem::path index_file_path = output_path / "index.json";
        std::ofstream index_file(index_file_path);
        index_file << index_data.dump(4) << std::endl;
        index_file.close();
    }
};

} // namespace MatSubset::Experiments

#endif // MAT_SUBSET_TESTER_H
