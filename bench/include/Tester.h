#ifndef MAT_SUBSET_BENCH_TESTER_H
#define MAT_SUBSET_BENCH_TESTER_H

#include <algorithm>  // For std::replace
#include <filesystem> // For std::filesystem::path
#include <functional> // For std::function
#include <iostream>   // For reading from and writing to the console
#include <map>        // For std::map
#include <memory>     // For std::unique_ptr
#include <ostream>    // For writing into files
#include <string>     // For std::string
#include <vector>     // For std::vector

#include <Eigen/Core>            // For vectors and matrices
#include <MatSubset/MatSubset.h> // For subset selection algorithms and utils
#include <nlohmann/json.hpp>     // For parsing .json files
#include <omp.h>                 // For parallelization

#include "MatrixGenerators.h" // For random matrix generators
#include "Utils.h"            // For add_underscores function

namespace MatSubset::Bench {

template <typename Scalar> class Tester {
  public:
    Tester(const nlohmann::json &experiments_config,
           const std::filesystem::path &output_path)
        : experiments_config(experiments_config), output_path(output_path) {

        populateSelectorFactory();
        populateMatrixGeneratorFactory();
    }

    void runAll() {
        for (auto &experiment_config : experiments_config) {
            runExperiment(experiment_config);
        }
    }

  private:
    const nlohmann::json experiments_config; //< Description of experiments
    const std::filesystem::path output_path; //< Path to the output directory

    // Factory maps and population methods for subset selectors
    using SelectorFactory =
        std::map<std::string,
                 std::function<std::unique_ptr<SelectorBase<Scalar>>(
                     const nlohmann::json &)>>;

    SelectorFactory selector_factory;

    template <typename SelectorType> void registerSelector() {
        auto dummy_instance = std::make_unique<SelectorType>();
        std::string name = dummy_instance->getAlgorithmName();
        selector_factory[name] = [](const nlohmann::json &selector_config) {
            return std::make_unique<SelectorType>();
        };
    }

    void populateSelectorFactory() {
        registerSelector<DualSetSelector<Scalar>>();
        registerSelector<FrobeniusRemovalSelector<Scalar>>();
        registerSelector<InterlacingFamiliesSelector<Scalar>>();
        registerSelector<RankRevealingQRSelector<Scalar>>();
        registerSelector<SelectorBase<Scalar>>();
        registerSelector<SpectralRemovalSelector<Scalar>>();
        registerSelector<SpectralSelectionSelector<Scalar>>();
        registerSelector<VolumeRemovalSelector<Scalar>>();
    }

    // Factory maps and population methods for subset random matrix
    // generators
    using MatrixGeneratorFactory =
        std::map<std::string,
                 std::function<std::unique_ptr<MatrixGeneratorBase<Scalar>>(
                     const nlohmann::json &)>>;

    MatrixGeneratorFactory matrix_generator_factory;

    template <typename MatrixGeneratorType> void registerMatrixGenerator() {
        if constexpr (std::is_same_v<SigmaMatrixGenerator<Scalar>,
                                     MatrixGeneratorType>) {
            Eigen::VectorX<Scalar> sigma_values(1);
            sigma_values << static_cast<Scalar>(1);
            auto dummy_instance =
                std::make_unique<MatrixGeneratorType>(1, 1, sigma_values);
            std::string name = dummy_instance->getMatrixType();
            matrix_generator_factory[name] =
                [](const nlohmann::json &generator_config) {
                    Eigen::Index m =
                        generator_config.at("rows").get<Eigen::Index>();
                    Eigen::Index n =
                        generator_config.at("cols").get<Eigen::Index>();

                    auto temp_vector = generator_config.at("singular_values")
                                           .get<std::vector<Scalar>>();

                    Eigen::VectorX<Scalar> sigma_values =
                        Eigen::Map<Eigen::VectorX<Scalar>>(temp_vector.data(),
                                                           temp_vector.size());

                    return std::make_unique<MatrixGeneratorType>(m, n,
                                                                 sigma_values);
                };
        } else {
            auto dummy_instance = std::make_unique<MatrixGeneratorType>(1, 1);
            std::string name = dummy_instance->getMatrixType();
            matrix_generator_factory[name] =
                [](const nlohmann::json &generator_config) {
                    Eigen::Index m =
                        generator_config.at("rows").get<Eigen::Index>();
                    Eigen::Index n =
                        generator_config.at("cols").get<Eigen::Index>();
                    return std::make_unique<MatrixGeneratorType>(m, n);
                };
        }
    }

    void populateMatrixGeneratorFactory() {
        registerMatrixGenerator<GaussianMatrixGenerator<Scalar>>();
        registerMatrixGenerator<GraphIncidenceMatrixGenerator<Scalar>>();
        registerMatrixGenerator<MatrixGeneratorBase<Scalar>>();
        registerMatrixGenerator<OrthonormalVectorsMatrixGenerator<Scalar>>();
        registerMatrixGenerator<SigmaMatrixGenerator<Scalar>>();
        registerMatrixGenerator<
            WeightedGraphIncidenceMatrixGenerator<Scalar>>();
    }

    // Function to run single experiment
    void runExperiment(const nlohmann::json &experiment_config) {
        // Check if the experiment is enabled
        if (!experiment_config.value("enabled", true)) {
            std::cout << "Skipping disabled experiment: "
                      << experiment_config.at("name").get<std::string>()
                      << std::endl;
            return;
        }

        // Construct the vector of relevant selectors
        std::vector<std::unique_ptr<SelectorBase<Scalar>>> selectors;
        const auto &algorithms_json = experiment_config.at("algorithms");

        for (const auto &algorithm_config : algorithms_json) {
            std::string algorithm_name =
                algorithm_config.at("name").get<std::string>();
            auto &factory_function = selector_factory.at(algorithm_name);
            auto selector_ptr = factory_function(algorithm_config);
            selectors.push_back(std::move(selector_ptr));
        }

        // Construct the matrix generator
        const auto &matrix_config = experiment_config.at("matrix");
        const std::string matrix_type =
            matrix_config.at("type").get<std::string>();
        auto &generator_factory_function =
            matrix_generator_factory.at(matrix_type);
        auto matrix_generator = generator_factory_function(matrix_config);

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

        // Create output files
        std::filesystem::create_directories(output_path);
        std::string experiment_name =
            experiment_config.at("name").get<std::string>();
        std::vector<std::ofstream> output_files;

        for (const auto &selector : selectors) {
            std::string algorithm_name = selector->getAlgorithmName();
            std::filesystem::path file_path =
                output_path / (Utils::add_underscores(experiment_name) + "_" +
                               Utils::add_underscores(algorithm_name) + ".csv");
            output_files.emplace_back(file_path);
            output_files.back() << "k,pinv_spectral_norm_ratio,pinv_frobenius_"
                                   "norm_ratio,wall_time_ms"
                                << std::endl;
        }

        // Main loop
        for (Eigen::Index k : k_values) {
#pragma omp parallel for schedule(static)
            for (int trial = 0; trial < trials_per_k; ++trial) {

                const Eigen::MatrixX<Scalar> X =
                    matrix_generator->generateMatrix();
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
                        output_files[i] << k << "," << pinv_spectral_norm_ratio
                                        << "," << pinv_frobenius_norm_ratio
                                        << "," << wall_time_ms << std::endl;
                    }
                }
            }
        }
    }
};

} // namespace MatSubset::Bench

#endif // MAT_SUBSET_TESTER_H
