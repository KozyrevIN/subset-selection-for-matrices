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
 *
 * The generation process involves:
 * 1. Creating an intermediate matrix with i.i.d. Gaussian entries.
 * 2. Performing a QR decomposition on this matrix.
 * 3. Correcting signs in R toi ensure all diagonal elements are nonnegative.
 * This ensures uniform (Haar) distribution.
 * 4. The resulting Q factor is the desired matrix with orthonormal columns.
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
        return "random matrix with orthonormal rows or columns";
    }

    /*!
     * @brief Generates the matrix with orthonormal vectors.
     * @return An Eigen::MatrixX<Scalar> of size m x n.
     *
     * If m >= n, the returned matrix has orthonormal columns.
     * If m < n, the returned matrix has orthonormal rows.
     */
    [[nodiscard]] Eigen::MatrixX<Scalar> generateMatrix() override {
        auto [initial_m, initial_n] = this->matrixSize;

        // The QR decomposition method requires the number of rows to be
        // greater than or equal to the number of columns to produce
        // orthonormal columns.
        bool needs_transpose = false;
        if (initial_m < initial_n) {
            needs_transpose = true;
        }

        // Generate the base Gaussian matrix to be orthogonalized.
        // We always create a "tall" matrix (rows >= cols).
        Eigen::MatrixX<Scalar> gaussian_matrix =
            (needs_transpose)
                ? this->generateGaussianMatrix(initial_n, initial_m)
                : this->generateGaussianMatrix(initial_m, initial_n);

        // Perform thin QR decomposition
        Eigen::HouseholderQR<Eigen::MatrixX<Scalar>> qr(gaussian_matrix);
        Eigen::MatrixX<Scalar> Q = qr.householderQ();

        // For a uniform (Haar) distribution, we must correct the signs of
        // the columns of Q based on the signs of the diagonal elements of R.
        auto R_diag = qr.matrixQR().diagonal();
        for (Eigen::Index j = 0; j < R_diag.size(); ++j) {
            if (R_diag(j) < 0) {
                Q.col(j) *= static_cast<Scalar>(-1);
            }
        }

        // If the original request was for a "wide" matrix (m < n), we
        // return the transpose, which will have orthonormal rows.
        if (needs_transpose) {
            return Q.transpose();
        } else {
            return Q;
        }
    }
};

} // namespace MatSubset::Bench

#endif // MAT_SUBSET_ORTHONORMAL_VECTORS_MATRIX_GENERATOR_H