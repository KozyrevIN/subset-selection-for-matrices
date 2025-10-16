#include <filesystem> // For std::filesystem::path
#include <fstream>    // For reading from and writing to files
#include <iostream>   // For reading from and writing to the console
#include <string>     // For std::string

#include <nlohmann/json.hpp> // For parsing .json files

#include "Tester.h" // For the Tester class

int main(int argc, char *argv[]) {
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
}
