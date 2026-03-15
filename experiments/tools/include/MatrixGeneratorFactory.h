#ifndef MAT_SUBSET_EXPERIMENTS_MATRIX_GENERATOR_FACTORY_H
#define MAT_SUBSET_EXPERIMENTS_MATRIX_GENERATOR_FACTORY_H

#include <functional> // For std::function
#include <map>        // For std::map
#include <memory>     // For std::unique_ptr
#include <random>     // For std::mt19937
#include <stdexcept>  // For std::runtime_error
#include <string>     // For std::string
#include <vector>     // For std::vector

#include <Eigen/Core>        // For Eigen::Index, Eigen::VectorX
#include <nlohmann/json.hpp> // For nlohmann::json

#include "MatrixGenerators.h" // For all matrix generator classes

namespace MatSubset::Experiments {

/*!
 * @brief Factory for creating matrix generator instances from JSON
 * configuration.
 * @tparam Scalar The underlying scalar type (e.g., float, double).
 *
 * This is a flexible base factory class that manages a registry of matrix
 * generator creation functions. Generators are identified by their matrix
 * type name and can be instantiated from JSON configuration objects.
 *
 * The factory uses a registration pattern where each generator type is
 * associated with a creation function (Creator) that takes a JSON config
 * and returns a unique_ptr to the generator instance.
 *
 * This class is designed to be extended (see DefaultMatrixGeneratorFactory)
 * or used directly with custom registrations for testing or experimentation.
 */
template <typename Scalar> class MatrixGeneratorFactory {
  public:
    /*!
     * @brief Default constructor.
     *
     * Creates an empty factory with no registered generators. Use
     * registerGenerator() to add generators, or use
     * DefaultMatrixGeneratorFactory which comes pre-populated with all
     * standard generators.
     */
    MatrixGeneratorFactory() = default;

    /*!
     * @brief Type alias for generator creation functions.
     *
     * A Creator is a callable that takes a JSON configuration object and
     * returns a unique_ptr to a newly created generator instance. The JSON
     * config typically contains "rows", "cols", and type-specific parameters.
     */
    using Creator = std::function<std::unique_ptr<MatrixGeneratorBase<Scalar>>(
        const nlohmann::json &)>;

    /*!
     * @brief Creates a matrix generator instance from JSON configuration.
     * @param config JSON object containing at least "type", "rows",
     * and "cols" fields. May contain additional parameters:
     *     - "seed" (optional): Fixed seed for reproducibility
     *     - "singular_values" (required for SigmaMatrixGenerator): Array of
     * singular values
     * @return A unique_ptr to the newly created generator instance.
     * @throws std::runtime_error if the generator type is not registered.
     * @throws nlohmann::json::exception if required fields are missing or
     * invalid.
     *
     * Example JSON configurations:
     * @code
     * // Random seed
     * {"type": "gaussian matrix", "rows": 10, "cols": 50}
     *
     * // Fixed seed for reproducibility
     * {"type": "gaussian matrix", "rows": 10, "cols": 50, "seed": 42}
     *
     * // SigmaMatrixGenerator with random seed
     * {"type": "matrix with a given set of singular values",
     *  "rows": 10, "cols": 50, "singular_values": [10.0, 5.0, 1.0]}
     *
     * // SigmaMatrixGenerator with fixed seed
     * {"type": "matrix with a given set of singular values",
     *  "rows": 10, "cols": 50, "singular_values": [10.0, 5.0, 1.0], "seed": 42}
     * @endcode
     */
    std::unique_ptr<MatrixGeneratorBase<Scalar>>
    create(const nlohmann::json &config) const {
        const std::string &type = config.at("type").get<std::string>();
        auto it = creators.find(type);
        if (it == creators.end()) {
            throw std::runtime_error("Unknown matrix generator type: " + type);
        }
        return it->second(config);
    }

    /*!
     * @brief Registers a new matrix generator type with the factory.
     * @param type The matrix type name (as returned by getMatrixType()).
     * @param creator A callable that creates generator instances from config.
     * @return true if registration succeeded, false if type already registered.
     *
     * This allows custom generators or alternative configurations to be
     * registered at runtime.
     *
     * Example:
     * @code
     * factory.registerGenerator("my matrix", [](const nlohmann::json& config) {
     *     Eigen::Index m = config.at("rows").get<Eigen::Index>();
     *     Eigen::Index n = config.at("cols").get<Eigen::Index>();
     *     return std::make_unique<MyCustomGenerator<double>>(m, n);
     * });
     * @endcode
     */
    bool registerGenerator(const std::string &type, Creator creator) {
        return creators.emplace(type, creator).second;
    }

  private:
    std::map<std::string, Creator>
        creators; ///< Registry of generator creators.
};

/*!
 * @brief Pre-configured factory with all standard MatSubset matrix generators.
 * @tparam Scalar The underlying scalar type (e.g., float, double).
 *
 * This factory comes pre-populated with all 6 standard matrix generator types
 * from the MatSubset experiments suite, ready to use. It handles different
 * constructor signatures automatically:
 *     - Standard generators taking (m, n): Most generators
 *     - SigmaMatrixGenerator taking (m, n, sigma_values): Special case
 *
 * Registered generators:
 *     - "zero matrix" - MatrixGeneratorBase (baseline, returns zeros)
 *     - "gaussian matrix" - GaussianMatrixGenerator
 *     - "graph incidence matrix" - GraphIncidenceMatrixGenerator
 *     - "weighted graph incidence matrix" -
 * WeightedGraphIncidenceMatrixGenerator
 *     - "matrix with orthonormal rows or columns" -
 * OrthonormalVectorsMatrixGenerator
 *     - "matrix with a given set of singular values" - SigmaMatrixGenerator
 *
 * Usage example:
 * @code
 * DefaultMatrixGeneratorFactory<double> factory;
 * nlohmann::json config = {
 *     {"type", "gaussian matrix"},
 *     {"rows", 10},
 *     {"cols", 50}
 * };
 * auto generator = factory.create(config);
 * auto X = generator->generateMatrix();
 * @endcode
 */
template <typename Scalar>
class DefaultMatrixGeneratorFactory : public MatrixGeneratorFactory<Scalar> {
  public:
    /*!
     * @brief Constructor that registers all standard matrix generators.
     *
     * Automatically populates the factory with creation functions for all
     * 6 standard MatSubset matrix generators. After construction, the
     * factory is immediately ready to create generators via create().
     */
    DefaultMatrixGeneratorFactory() {
        registerStandardGenerator<MatrixGeneratorBase>();
        registerStandardGenerator<GaussianMatrixGenerator>();
        registerStandardGenerator<GraphIncidenceMatrixGenerator>();
        registerStandardGenerator<WeightedGraphIncidenceMatrixGenerator>();
        registerStandardGenerator<OrthonormalVectorsMatrixGenerator>();
        registerSigmaGenerator();
        registerMatrixFromFileGenerator();
    }

  private:
    /*!
     * @brief Registers a standard generator with (m, n) or (m, n, seed) constructor.
     * @tparam Generator Template template parameter for the generator class.
     *
     * Creates a registration for generators that take (rows, cols) or
     * (rows, cols, seed) in their constructor. The generator's matrix type is
     * automatically retrieved via getMatrixType(). If the JSON config contains
     * an optional "seed" field, it's passed to the constructor for reproducibility;
     * otherwise, random seeding is used.
     */
    template <template <typename> class Generator>
    void registerStandardGenerator() {
        auto dummy = std::make_unique<Generator<Scalar>>(1, 1);
        std::string type = dummy->getMatrixType();
        typename MatrixGeneratorFactory<Scalar>::Creator creator =
            [](const nlohmann::json &config) {
                Eigen::Index m = config.at("rows").get<Eigen::Index>();
                Eigen::Index n = config.at("cols").get<Eigen::Index>();

                if (config.contains("seed")) {
                    auto seed =
                        config.at("seed").get<std::mt19937::result_type>();
                    return std::make_unique<Generator<Scalar>>(m, n, seed);
                } else {
                    return std::make_unique<Generator<Scalar>>(m, n);
                }
            };
        this->registerGenerator(type, creator);
    }

    /*!
     * @brief Registers SigmaMatrixGenerator with required singular_values parameter.
     *
     * SigmaMatrixGenerator requires an additional "singular_values" parameter
     * (array of singular values where k <= min(m, n)). If the JSON config contains
     * an optional "seed" field, it's passed to the constructor for reproducibility;
     * otherwise, random seeding is used.
     *
     * @note Uses dummy sigma values for registration only; actual instances
     * use the sigma_values from config.
     */
    void registerSigmaGenerator() {
        Eigen::VectorX<Scalar> dummy_sigma(1);
        dummy_sigma << static_cast<Scalar>(1);
        auto dummy =
            std::make_unique<SigmaMatrixGenerator<Scalar>>(1, 1, dummy_sigma);
        std::string type = dummy->getMatrixType();

        typename MatrixGeneratorFactory<Scalar>::Creator creator =
            [](const nlohmann::json &config) {
                Eigen::Index m = config.at("rows").get<Eigen::Index>();
                Eigen::Index n = config.at("cols").get<Eigen::Index>();

                auto sigma_vec =
                    config.at("singular_values").get<std::vector<Scalar>>();
                Eigen::VectorX<Scalar> sigma_values =
                    Eigen::Map<Eigen::VectorX<Scalar>>(sigma_vec.data(),
                                                       sigma_vec.size());

                if (config.contains("seed")) {
                    auto seed =
                        config.at("seed").get<std::mt19937::result_type>();
                    return std::make_unique<SigmaMatrixGenerator<Scalar>>(
                        m, n, seed, sigma_values);
                } else {
                    return std::make_unique<SigmaMatrixGenerator<Scalar>>(
                        m, n, sigma_values);
                }
            };
        this->registerGenerator(type, creator);
    }

    /*!
     * @brief Registers MatrixFromFileGenerator with required file_path parameter.
     *
     * MatrixFromFileGenerator requires a "file_path" parameter specifying the
     * path to the data file (ARFF, CSV, or other supported formats). The matrix
     * dimensions are automatically determined from the file contents.
     *
     * Example JSON configuration:
     * @code
     * {"type": "matrix from file",
     *  "file_path": "supplementary/bank32nh.arff"}
     * @endcode
     *
     * @note The "rows" and "cols" fields are not required for this generator
     * as dimensions are determined from the file.
     */
    void registerMatrixFromFileGenerator() {
        // Use the known type string directly instead of creating a dummy instance
        std::string type = "matrix from file";

        typename MatrixGeneratorFactory<Scalar>::Creator creator =
            [](const nlohmann::json &config) {
                std::string file_path = config.at("file_path").get<std::string>();
                return std::make_unique<MatrixFromFileGenerator<Scalar>>(file_path);
            };
        this->registerGenerator(type, creator);
    }
};

} // namespace MatSubset::Experiments

#endif // MAT_SUBSET_EXPERIMENTS_MATRIX_GENERATOR_FACTORY_H
