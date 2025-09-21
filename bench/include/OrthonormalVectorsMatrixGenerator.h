#ifndef MAT_SUBSET_ORTHONORMAL_VECTORS_MATRIX_GENERATOR_H
#define MAT_SUBSET_ORTHONORMAL_VECTORS_MATRIX_GENERATOR_H

#include <cmath> // For std::abs

#include <Eigen/QR> // For Eigen::HouseholderQR

#include "GaussianMatrixGenerator.h" // Base class

namespace MatSubset::Bench {

/*!
 * @brief Generates a random matrix with orthonormal columns or rows.
 * @tparam Scalar The underlying scalar type of the matrix elements (e.g.,
 * float, double).
 *
 * This class generates a matrix that is uniformly distributed over the Stiefel
 * manifold V_k(R^n), meaning its columns (if rows >= cols) or rows (if rows <
 * cols) form an orthonormal set.
 */
template <typename Scalar>
class OrthonormalVectorsMatrixGenerator
    : public GaussianMatrixGenerator<Scalar> {
  public:
    /*!
     * @brief Constructor with a random seed.
     * @param m Number of rows.
     * @param n Number of columns.
     */
    OrthonormalVectorsMatrixGenerator(Eigen::Index m, Eigen::Index n)
        : GaussianMatrixGenerator<Scalar>(m, n) {}

    /*!
     * @brief Constructor with a specified seed.
     * @param m Number of rows.
     * @param n Number of columns.
     * @param seed Seed for the random number generator.
     */
    OrthonormalVectorsMatrixGenerator(Eigen::Index m, Eigen::Index n,
                                      std::mt19937::result_type seed)
        : GaussianMatrixGenerator<Scalar>(m, n, seed) {}

    /*!
     * @brief Gets a string description of the matrix type.
     * @return A string identifying the matrix as having orthonormal
     * rows/columns.
     */
    [[nodiscard]] std::string getMatrixType() const override {
        return "matrix with orthonormal rows or columns";
    }

    /*!
     * @brief Generates the matrix with orthonormal vectors using
     * constructor-defined dimensions.
     */
    [[nodiscard]] Eigen::MatrixX<Scalar> generateMatrix() override {
        auto [m, n] = this->matrixSize;
        return generateOrthonormalMatrix(m, n);
    }

  protected:
    /*!
     * @brief Generates a random matrix of arbitrary size with orthonormal
     * columns or rows.
     * @param rows The number of rows for the new matrix.
     * @param cols The number of columns for the new matrix.
     * @return An Eigen::MatrixX<Scalar> of size rows x cols.
     *
     * The generation process involves:
     * 1. Creating an intermediate matrix with i.i.d. Gaussian entries.
     * 2. Performing a QR decomposition on this matrix.
     * 3. Correcting signs in R toi ensure all diagonal elements are
     * nonnegative. This ensures uniform (Haar) distribution.
     * 4. The resulting Q factor is the desired matrix with orthonormal columns.
     *
     * This helper is exposed to derived classes that need to generate
     * random orthonormal matrices as part of a larger algorithm.
     */
    [[nodiscard]] Eigen::MatrixX<Scalar>
    generateOrthonormalMatrix(Eigen::Index rows, Eigen::Index cols) {
        bool needs_transpose = (rows < cols);

        Eigen::Index tall_rows = needs_transpose ? cols : rows;
        Eigen::Index tall_cols = needs_transpose ? rows : cols;

        Eigen::MatrixX<Scalar> gaussian_matrix =
            this->generateGaussianMatrix(tall_rows, tall_cols);

        Eigen::HouseholderQR<Eigen::MatrixX<Scalar>> qr(gaussian_matrix);
        Eigen::MatrixX<Scalar> Q =
            qr.householderQ() *
            Eigen::MatrixX<Scalar>::Identity(tall_rows, tall_cols);

        Eigen::VectorX<Scalar> R_diag = qr.matrixQR().diagonal();
        for (Eigen::Index j = 0; j < R_diag.size(); ++j) {
            if (R_diag(j) < 0) {
                Q.col(j) *= static_cast<Scalar>(-1);
            }
        }

        return needs_transpose ? Q.transpose() : Q;
    }
};

} // namespace MatSubset::Bench

#endif // MAT_SUBSET_ORTHONORMAL_VECTORS_MATRIX_GENERATOR_H