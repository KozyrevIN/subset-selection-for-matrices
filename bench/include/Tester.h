#ifndef MAT_SUBSET_TESTER_H
#define MAT_SUBSET_TESTER_H

#include <filesystem> // For std::filesystem::path
#include <functional> // For std::function
#include <iostream>   // For reading from and writing to the console
#include <map>        // For std::map
#include <memory>     // For std::unique_ptr
#include <string>     // For std::string
#include <vector>     // For std::vector

#include <Eigen/Core>               // For vectors and matrices
#include <MatSubset/SelectorBase.h> // For subset selection algorithms
#include <nlohmann/json.hpp>        // For parsing .json files

#include "MatrixGenerator.h" // For random matrix generators

namespace MatSubset::Bench {

template <typename Scalar> class Tester {
  public:
    Tester(const nlohmann::json &experiments_config,
           const std::filesystem::path &output_path)
        : experiments_config(experiments_config), output_path(output_path) {
        std::cout << std::filesystem::absolute(output_path) << std::endl;
    }
    void runAll() {}

  private:
    const nlohmann::json experiments_config; //< Description of experiments
    const std::filesystem::path output_path; //< Path to the output directory

    // Internal struct to hold results for a single value of k
    struct Result {
        Eigen::Index k;
        Scalar spectral_pinv_norm;
        Scalar frobenius_pinv_norm;
        double average_time_ms;
    };

    // Factory maps and population methods
    using SelectorFactory =
        std::map<std::string,
                 std::function<std::unique_ptr<SelectorBase<Scalar>>(
                     const nlohmann::json &)>>;
    using MatrixGeneratorFactory =
        std::map<std::string,
                 std::function<std::unique_ptr<MatrixGenerator<Scalar>>(
                     const nlohmann::json &)>>;

    SelectorFactory selector_factory;
    MatrixGeneratorFactory matrix_generator_factory;

    void populateSelectorFactory();
    void populateMatrixFactory();

    // Factory access methods
    std::unique_ptr<SelectorBase<double>>
    createSelector(const nlohmann::json &selector_config);
    std::unique_ptr<MatrixGenerator<double>>
    createMatrixGenerator(const nlohmann::json &matrix_config);

    // Helper to write data to a CSV file
    void writeResults(const std::string &file_path,
                      const std::vector<Result> &results);

    // Function to run single experiment
    void runExperiment(const nlohmann::json &experiment_config);
};

} // namespace MatSubset::Bench

#endif // MAT_SUBSET_TESTER_H
