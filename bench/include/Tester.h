#ifndef MAT_SUBSET_BENCH_TESTER_H
#define MAT_SUBSET_BENCH_TESTER_H

#include <chrono>     // For timestamps
#include <ctime>      // For std::ctime
#include <filesystem> // For std::filesystem::path
#include <format>     // For std::format
#include <fstream>    // For std::ofstream
#include <iostream>   // For reading from and writing to the console
#include <memory>     // For std::unique_ptr
#include <string>     // For std::string
#include <vector>     // For std::vector

#include <Eigen/Core>            // For vectors and matrices
#include <MatSubset/MatSubset.h> // For subset selection algorithms and utils
#include <nlohmann/json.hpp>     // For parsing .json files
#include <omp.h>                 // For parallelization

#include "MatrixGeneratorFactory.h" // For matrix generator factory
#include "ProgressBar.h"            // For progress bar
#include "SelectorFactory.h"        // For selector factory
#include "Utils.h"                  // For add_underscores function

namespace MatSubset::Bench {

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

        // Construct the vector of relevant selectors
        std::vector<std::unique_ptr<SelectorBase<Scalar>>> selectors;
        const auto &algorithms_json = experiment_config.at("algorithms");

        for (const auto &algorithm_config : algorithms_json) {
            auto selector_ptr = selector_factory.create(algorithm_config);
            selectors.push_back(std::move(selector_ptr));
        }

        // Construct the matrix generator
        const auto &matrix_config = experiment_config.at("matrix");
        auto matrix_generator = matrix_generator_factory.create(matrix_config);

        // Determine k values
        std::vector<Eigen::Index> k_values;
        if (experiment_config.contains("k_values")) {
            k_values = experiment_config.at("k_values")
                           .get<std::vector<Eigen::Index>>();
        } else if (experiment_config.contains("k_values_range")) {
            const auto &range = experiment_config.at("k_values_range");
            Eigen::Index start = range.at("start").get<Eigen::Index>();
            Eigen::Index stop = range.at("stop").get<Eigen::Index>();
            Eigen::Index step = range.at("step").get<Eigen::Index>();
            for (Eigen::Index k = start; k <= stop; k += step) {
                k_values.push_back(k);
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

        // Replace k_values_range with the actual k_values array if it was a
        // range
        if (saved_config.contains("k_values_range")) {
            saved_config.erase("k_values_range");
        }
        saved_config["k_values"] = k_values;

        // Remove runtime-specific fields
        saved_config.erase("enabled");

        // Add metadata with start time
        auto start_time = std::chrono::system_clock::now();
        auto start_timestamp = std::chrono::system_clock::to_time_t(start_time);
        saved_config["metadata"]["start_time"] = std::ctime(&start_timestamp);
        saved_config["metadata"]["num_threads"] = omp_get_max_threads();

        // Write initial config (will be updated with finish time later)
        std::filesystem::path config_file_path =
            experiment_folder / "config.json";

        std::vector<std::ofstream> output_files;

        for (const auto &selector : selectors) {
            std::string algorithm_name = selector->getAlgorithmName();
            std::filesystem::path file_path =
                experiment_folder /
                (Utils::add_underscores(algorithm_name) + ".csv");
            output_files.emplace_back(file_path);
            output_files.back()
                << "k,pinv_spectral_norm_ratio,pinv_frobenius_"
                   "norm_ratio,wall_time_ms,spectral_bound,frobenius_bound"
                << std::endl;
        }

        // Main loop with progress tracking
        std::cout << std::endl; // Add newline so progress bar appears below
        ProgressBar progress_bar(k_values.size() * trials_per_k *
                                 selectors.size());
        int k_max_len = std::to_string(k_values.back()).length();

        for (Eigen::Index k : k_values) {
            // Calculate theoretical bounds once per k (independent of trials)
            auto [m, n] = matrix_generator->getMatrixSize();

            std::vector<Scalar> spectral_bounds(selectors.size());
            std::vector<Scalar> frobenius_bounds(selectors.size());

            for (int i = 0; i < selectors.size(); ++i) {
                spectral_bounds[i] = std::sqrt(
                    selectors[i]->template bound<Norm::Spectral>(m, n, k));
                frobenius_bounds[i] = std::sqrt(
                    selectors[i]->template bound<Norm::Frobenius>(m, n, k));
            }

#pragma omp parallel for schedule(static)
            for (int trial = 0; trial < trials_per_k; ++trial) {
                
                // generateNatrix() is thread safe
                Eigen::MatrixX<Scalar> X = matrix_generator->generateMatrix();

                Scalar X_dag_spectral_norm =
                    MatSubset::Utils::pinv_norm<Scalar, Norm::Spectral>(X);
                Scalar X_dag_frobenius_norm =
                    MatSubset::Utils::pinv_norm<Scalar, Norm::Frobenius>(X);

                for (int i = 0; i < selectors.size(); ++i) {
                    auto start_time = omp_get_wtime();
                    std::vector<Eigen::Index> selected_indices =
                        selectors[i]->selectSubset(X, k);
                    auto end_time = omp_get_wtime();

                    double wall_time_ms = (end_time - start_time) * 1000.0;

                    Eigen::MatrixX<Scalar> X_S =
                        X(Eigen::all, selected_indices);
                    Scalar X_S_dag_spectral_norm =
                        MatSubset::Utils::pinv_norm<Scalar, Norm::Spectral>(
                            X_S);
                    Scalar X_S_dag_frobenius_norm =
                        MatSubset::Utils::pinv_norm<Scalar, Norm::Frobenius>(
                            X_S);
                    Scalar pinv_spectral_norm_ratio =
                        X_dag_spectral_norm / X_S_dag_spectral_norm;
                    Scalar pinv_frobenius_norm_ratio =
                        X_dag_frobenius_norm / X_S_dag_frobenius_norm;
#pragma omp critical
                    {
                        std::string label =
                            "k = " + std::format("{:^{}}", k, k_max_len);
                        //progress_bar.update(label);
                        output_files[i] << k << "," << pinv_spectral_norm_ratio
                                        << "," << pinv_frobenius_norm_ratio
                                        << "," << wall_time_ms << ","
                                        << spectral_bounds[i] << ","
                                        << frobenius_bounds[i] << std::endl;
                    }
                }
            }
        }

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

} // namespace MatSubset::Bench

#endif // MAT_SUBSET_TESTER_H
