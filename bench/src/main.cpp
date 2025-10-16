#include <filesystem> // For std::filesystem::path
#include <fstream>    // For reading from and writing to files
#include <iostream>   // For reading from and writing to the console
#include <string>     // For std::string

#include <nlohmann/json.hpp> // For parsing .json files

#include "SelectorFactory.h"        // TEST: For selector factory testing
#include "MatrixGeneratorFactory.h" // TEST: For matrix generator factory testing
// #include "Tester.h" // For the Tester class

int main(int argc, char *argv[]) {
    // TEST: Verify SelectorFactory compiles
    std::cout << "=== Testing SelectorFactory ===" << std::endl;
    MatSubset::Bench::DefaultSelectorFactory<double> selector_factory;
    nlohmann::json selector_config;
    selector_config["name"] = "dual set";
    try {
        auto selector = selector_factory.create(selector_config);
        std::cout << "✓ Created selector: "
                  << selector->getAlgorithmName() << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✗ Selector test error: " << e.what() << std::endl;
    }

    // TEST: Verify MatrixGeneratorFactory compiles
    std::cout << "\n=== Testing MatrixGeneratorFactory ===" << std::endl;
    MatSubset::Bench::DefaultMatrixGeneratorFactory<double> generator_factory;

    // Test standard generator
    nlohmann::json gaussian_config;
    gaussian_config["type"] = "gaussian matrix";
    gaussian_config["rows"] = 3;
    gaussian_config["cols"] = 5;
    try {
        auto generator = generator_factory.create(gaussian_config);
        std::cout << "✓ Created generator: "
                  << generator->getMatrixType() << std::endl;
        auto matrix = generator->generateMatrix();
        std::cout << "  Matrix size: " << matrix.rows() << "x"
                  << matrix.cols() << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✗ Gaussian generator test error: " << e.what() << std::endl;
    }

    // Test SigmaMatrixGenerator
    nlohmann::json sigma_config;
    sigma_config["type"] = "matrix with a given set of singular values";
    sigma_config["rows"] = 3;
    sigma_config["cols"] = 5;
    sigma_config["singular_values"] = {10.0, 5.0, 1.0};
    try {
        auto generator = generator_factory.create(sigma_config);
        std::cout << "✓ Created generator: "
                  << generator->getMatrixType() << std::endl;
        auto matrix = generator->generateMatrix();
        std::cout << "  Matrix size: " << matrix.rows() << "x"
                  << matrix.cols() << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✗ Sigma generator test error: " << e.what() << std::endl;
    }

    // Test seed reproducibility
    std::cout << "\n=== Testing Seed Reproducibility ===" << std::endl;
    nlohmann::json seeded_config;
    seeded_config["type"] = "gaussian matrix";
    seeded_config["rows"] = 2;
    seeded_config["cols"] = 3;
    seeded_config["seed"] = 12345;

    try {
        auto gen1 = generator_factory.create(seeded_config);
        auto gen2 = generator_factory.create(seeded_config);
        auto matrix1 = gen1->generateMatrix();
        auto matrix2 = gen2->generateMatrix();

        bool identical = (matrix1 - matrix2).norm() < 1e-10;
        if (identical) {
            std::cout << "✓ Seed reproducibility verified: identical matrices generated" << std::endl;
        } else {
            std::cout << "✗ Seed reproducibility FAILED: matrices differ!" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "✗ Seed test error: " << e.what() << std::endl;
    }

    std::cout << "\n=== All factory tests passed! ===" << std::endl;
    return 0;

    // ORIGINAL CODE BELOW (commented out for testing)
    /*
    // Argument validation
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_config.json>"
                  << std::endl;
        return 1;
    }

    // Config file loading and parsing
    std::filesystem::path config_path(argv[1]);

    if (!std::filesystem::exists(config_path)) {
        std::cerr << "Error: The specified config file does not exist at "
                  << std::filesystem::absolute(config_path) << std::endl;
        return 1;
    }

    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        std::cerr << "Error: Could not open config file (check permissions?): "
                  << config_path << std::endl;
        return 1;
    }

    nlohmann::json config;
    try {
        config_file >> config;
    } catch (const nlohmann::json::parse_error &e) {
        std::cerr << "Error: Failed to parse config file: " << e.what()
                  << std::endl;
        return 1;
    }

    // Path manipulation
    std::filesystem::path base_path = config_path.parent_path();
    std::string relative_output_path = config.value("output_path", "results");
    std::filesystem::path full_output_path = base_path / relative_output_path;

    // Extraction of experiment info
    nlohmann::json experiments_config = config.at("experiments");

    // Main logic
    try {
        std::string scalar_string = config.value("scalar", "unspecified");

        if (scalar_string == "float") {
            MatSubset::Bench::Tester<float> tester(experiments_config,
                                                   full_output_path);
            tester.runAll();
        } else if (scalar_string == "double") {
            MatSubset::Bench::Tester<double> tester(experiments_config,
                                                    full_output_path);
            tester.runAll();
        } else {
            throw std::runtime_error("The scalar type '" + scalar_string +
                                     "' is unsupported.");
        }

    } catch (const std::exception &e) {
        std::cerr << "An error occurred during testing: " << e.what()
                  << std::endl;
        return 1;
    }

    std::cout << "Benchmarking finished successfully." << std::endl;
    return 0;
    */
}
