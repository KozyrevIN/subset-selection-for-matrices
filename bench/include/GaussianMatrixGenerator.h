#ifndef MAT_SUBSET_GAUSSIAN_MATRIX_GENERATOR_H
#define MAT_SUBSET_GAUSSIAN_MATRIX_GENERATOR_H

#include "MatrixGenerator.h" // For the base class

namespace MatSubset::Bench {

/*!
 * @brief Generates a matrix with i.i.d. Gaussian (normal) entries.
 * @tparam Scalar The underlying scalar type of the matrix elements (e.g.,
 * float, double).
 *
 * This class derives from MatrixGenerator and specializes in creating matrices
 * where each element is drawn independently from a standard normal
 * distribution (mean 0, variance 1).
 */
template <typename Scalar>
class GaussianMatrixGenerator : public MatrixGenerator<Scalar> {
  public:
    /*!
     * @brief Constructor for GaussianMatrixGenerator with a random seed.
     * @param m Number of rows.
     * @param n Number of columns.
     *
     * Inherits the base class constructor that uses a high-quality random seed.
     */
    GaussianMatrixGenerator(Eigen::Index m, Eigen::Index n)
        : MatrixGenerator<Scalar>(m, n) {}

    /*!
     * @brief Constructor for GaussianMatrixGenerator with a specified seed.
     * @param m Number of rows.
     * @param n Number of columns.
     * @param seed Seed for the random number generator.
     */
    GaussianMatrixGenerator(Eigen::Index m, Eigen::Index n,
                            std::mt19937::result_type seed)
        : MatrixGenerator<Scalar>(m, n, seed) {}

    /*!
     * @brief Gets a string description of the matrix type.
     * @return A string identifying the matrix as having i.i.d. Gaussian
     * entries.
     */
    [[nodiscard]] std::string getMatrixType() const override {
        return "random matrix with i.i.d. gaussian entries";
    }

    /*!
     * @brief Generates a matrix with entries from a standard normal
     * distribution.
     *
     * This method uses the dimensions specified in the constructor.
     */
    [[nodiscard]] Eigen::MatrixX<Scalar> generateMatrix() override {
        auto [m, n] = this->matrixSize;
        return generateGaussianMatrix(m, n);
    }

  protected:
    /*!
     * @brief Generates a Gaussian matrix of arbitrary size.
     * @param rows The number of rows for the new matrix.
     * @param cols The number of columns for the new matrix.
     * @return An Eigen::MatrixX<Scalar> of size rows x cols with i.i.d. N(0,1)
     * entries.
     *
     * This helper function is exposed to derived classes so they can use the
     * Gaussian generation mechanism for their own purposes, using the same
     * seeded random number generator.
     */
    [[nodiscard]] Eigen::MatrixX<Scalar>
    generateGaussianMatrix(Eigen::Index rows, Eigen::Index cols) {
        Eigen::MatrixX<Scalar> mat(rows, cols);
        std::normal_distribution<Scalar> dist(0.0, 1.0);
        for (Eigen::Index i = 0; i < rows; ++i) {
            for (Eigen::Index j = 0; j < cols; ++j) {
                mat(i, j) = dist(this->gen);
            }
        }
        return mat;
    }
};

} // namespace MatSubset::Bench

#endif // MAT_SUBSET_GAUSSIAN_MATRIX_GENERATOR_H