#ifndef MAT_SUBSET_EXPERIMENTS_MATRIX_FROM_FILE_GENERATOR_H
#define MAT_SUBSET_EXPERIMENTS_MATRIX_FROM_FILE_GENERATOR_H

#include <algorithm>  // For std::find, std::remove_if
#include <cctype>     // For std::isspace
#include <filesystem> // For std::filesystem::path
#include <fstream>    // For std::ifstream
#include <sstream>    // For std::stringstream
#include <stdexcept>  // For std::runtime_error
#include <string>     // For std::string
#include <vector>     // For std::vector

#include "MatrixGeneratorBase.h" // For the base class

namespace MatSubset::Experiments {

/*!
 * @brief Generates a matrix by reading data from a file.
 * @tparam Scalar The underlying scalar type of the matrix elements (e.g.,
 * float, double).
 *
 * This class derives from MatrixGeneratorBase and reads matrix data from
 * files in the supplementary folder. Supported formats include:
 * - ARFF (Attribute-Relation File Format): Commonly used for datasets
 * - CSV (Comma-Separated Values): Simple tabular data format
 *
 * The matrix is loaded once during construction and cached for subsequent
 * generateMatrix() calls, making it efficient for repeated usage.
 *
 * @note Unlike other generators, this class does not use the RNG from the
 * base class since it returns the same matrix data from the file each time.
 */
template <typename Scalar>
class MatrixFromFileGenerator : public MatrixGeneratorBase<Scalar> {
  private:
    Eigen::MatrixX<Scalar> cached_matrix; ///< Cached matrix data from file.
    std::string file_path; ///< Path to the source file.

  public:
    /*!
     * @brief Constructor that loads matrix data from a file.
     * @param file_path Path to the data file (relative or absolute).
     *
     * The file format is automatically detected from the extension:
     * - .arff: ARFF format
     * - .csv or .data: CSV format
     *
     * The matrix dimensions are determined from the loaded data.
     * If the loaded matrix is tall (rows > cols), it is automatically
     * transposed to be wide (rows <= cols), as required by the subset
     * selection algorithms.
     *
     * @throws std::runtime_error if the file cannot be read or parsed.
     */
    MatrixFromFileGenerator(const std::string &file_path)
        : MatrixGeneratorBase<Scalar>(0, 0), file_path(file_path) {
        loadMatrixFromFile();

        // Transpose if necessary to ensure wide matrix (m <= n)
        if (cached_matrix.rows() > cached_matrix.cols()) {
            cached_matrix.transposeInPlace();
        }

        // Update the matrixSize after loading and potential transpose
        const_cast<std::pair<Eigen::Index, Eigen::Index> &>(this->matrixSize) =
            {cached_matrix.rows(), cached_matrix.cols()};
    }

    /*!
     * @brief Gets a string description of the matrix type.
     * @return A string identifying this as a matrix loaded from a file.
     */
    [[nodiscard]] std::string getMatrixType() const override {
        return "matrix from file";
    }

    /*!
     * @brief Generates (returns) the matrix loaded from the file.
     * @return A copy of the matrix data loaded from the file.
     *
     * This method returns a copy of the cached matrix, ensuring thread-safe
     * usage even though the same data is returned each time.
     */
    [[nodiscard]] Eigen::MatrixX<Scalar> generateMatrix() override {
        // Return a copy of the cached matrix
        // Lock mutex for thread-safety (even though we're just copying)
        std::lock_guard<std::mutex> lock(this->gen_mutex);
        return cached_matrix;
    }

    /*!
     * @brief Gets the path to the source file.
     * @return The file path used to load the matrix.
     */
    [[nodiscard]] std::string getFilePath() const { return file_path; }

  private:
    /*!
     * @brief Loads matrix data from the file.
     *
     * Automatically detects the file format based on extension and parses
     * the file accordingly.
     *
     * @throws std::runtime_error if the file cannot be opened or parsed.
     */
    void loadMatrixFromFile() {
        std::filesystem::path path(file_path);
        std::string extension = path.extension().string();

        // Convert extension to lowercase for comparison
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char c) { return std::tolower(c); });

        if (extension == ".arff") {
            loadARFF();
        } else if (extension == ".csv" || extension == ".data") {
            loadCSV(',');
        } else {
            throw std::runtime_error("Unsupported file format: " + extension +
                                     ". Supported formats: .arff, .csv, .data");
        }
    }

    /*!
     * @brief Parses an ARFF file and loads the matrix data.
     *
     * ARFF format consists of:
     * - Comments (lines starting with %)
     * - @relation line
     * - @attribute lines defining each column
     * - @data line followed by comma-separated numerical data
     *
     * @throws std::runtime_error if the file cannot be read or has invalid
     * format.
     */
    void loadARFF() {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + file_path);
        }

        std::vector<std::vector<Scalar>> data;
        std::string line;
        bool in_data_section = false;
        int num_attributes = 0;

        while (std::getline(file, line)) {
            // Trim whitespace
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);

            // Skip empty lines
            if (line.empty()) {
                continue;
            }

            // Convert to lowercase for comparison
            std::string line_lower = line;
            std::transform(line_lower.begin(), line_lower.end(),
                           line_lower.begin(),
                           [](unsigned char c) { return std::tolower(c); });

            // Skip comments
            if (line[0] == '%') {
                continue;
            }

            // Count attributes
            if (line_lower.find("@attribute") == 0) {
                num_attributes++;
                continue;
            }

            // Check for @data section
            if (line_lower.find("@data") == 0) {
                in_data_section = true;
                continue;
            }

            // Skip other @-directives
            if (line[0] == '@') {
                continue;
            }

            // Parse data lines
            if (in_data_section) {
                std::vector<Scalar> row;
                std::stringstream ss(line);
                std::string value;

                while (std::getline(ss, value, ',')) {
                    // Trim whitespace from value
                    value.erase(0, value.find_first_not_of(" \t"));
                    value.erase(value.find_last_not_of(" \t") + 1);

                    if (!value.empty()) {
                        try {
                            // Handle missing values (represented as '?')
                            if (value == "?") {
                                row.push_back(static_cast<Scalar>(0));
                            } else {
                                row.push_back(static_cast<Scalar>(
                                    std::stod(value)));
                            }
                        } catch (const std::exception &) {
                            // Try to parse categorical data as numeric, or
                            // skip
                            row.push_back(static_cast<Scalar>(0));
                        }
                    }
                }

                if (!row.empty()) {
                    data.push_back(row);
                }
            }
        }

        file.close();

        if (data.empty()) {
            throw std::runtime_error("No data found in ARFF file: " +
                                     file_path);
        }

        // Convert vector of vectors to Eigen matrix
        Eigen::Index rows = data.size();
        Eigen::Index cols = data[0].size();

        cached_matrix.resize(rows, cols);
        for (Eigen::Index i = 0; i < rows; ++i) {
            if (static_cast<Eigen::Index>(data[i].size()) != cols) {
                throw std::runtime_error(
                    "Inconsistent number of columns in ARFF file at row " +
                    std::to_string(i));
            }
            for (Eigen::Index j = 0; j < cols; ++j) {
                cached_matrix(i, j) = data[i][j];
            }
        }
    }

    /*!
     * @brief Parses a CSV file and loads the matrix data.
     * @param delimiter The delimiter character (typically ',').
     *
     * CSV format is simply rows of comma-separated numerical values.
     * Lines starting with '#' are treated as comments.
     *
     * @throws std::runtime_error if the file cannot be read or has invalid
     * format.
     */
    void loadCSV(char delimiter = ',') {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + file_path);
        }

        std::vector<std::vector<Scalar>> data;
        std::string line;

        while (std::getline(file, line)) {
            // Trim whitespace
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);

            // Skip empty lines and comments
            if (line.empty() || line[0] == '#') {
                continue;
            }

            std::vector<Scalar> row;
            std::stringstream ss(line);
            std::string value;

            while (std::getline(ss, value, delimiter)) {
                // Trim whitespace from value
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);

                if (!value.empty()) {
                    try {
                        // Check if the value is a number
                        bool is_number = true;
                        for (char c : value) {
                            if (!std::isdigit(c) && c != '.' && c != '-' &&
                                c != '+' && c != 'e' && c != 'E') {
                                is_number = false;
                                break;
                            }
                        }

                        if (is_number) {
                            row.push_back(
                                static_cast<Scalar>(std::stod(value)));
                        } else {
                            // Skip non-numeric columns (like categorical data
                            // in first column)
                            continue;
                        }
                    } catch (const std::exception &) {
                        // Skip invalid values
                        continue;
                    }
                }
            }

            if (!row.empty()) {
                data.push_back(row);
            }
        }

        file.close();

        if (data.empty()) {
            throw std::runtime_error("No data found in CSV file: " + file_path);
        }

        // Convert vector of vectors to Eigen matrix
        Eigen::Index rows = data.size();
        Eigen::Index cols = data[0].size();

        cached_matrix.resize(rows, cols);
        for (Eigen::Index i = 0; i < rows; ++i) {
            if (static_cast<Eigen::Index>(data[i].size()) != cols) {
                throw std::runtime_error(
                    "Inconsistent number of columns in CSV file at row " +
                    std::to_string(i));
            }
            for (Eigen::Index j = 0; j < cols; ++j) {
                cached_matrix(i, j) = data[i][j];
            }
        }
    }
};

} // namespace MatSubset::Experiments

#endif // MAT_SUBSET_EXPERIMENTS_MATRIX_FROM_FILE_GENERATOR_H
