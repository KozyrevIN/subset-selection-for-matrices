#ifndef MAT_SUBSET_BENCH_SIGMA_MATRIX_GENERATOR_H
#define MAT_SUBSET_BENCH_SIGMA_MATRIX_GENERATOR_H

#include <algorithm> // For std::min
#include <cassert>   // For assert

#include "OrthonormalVectorsMatrixGenerator.h" // Base class

namespace MatSubset::Bench {

/*!
 * @brief Generates a random matrix with a predefined set of singular values.
 * @tparam Scalar The underlying scalar type of the matrix elements.
 *
 * This class constructs a matrix A = U * Sigma * V^T, where U and V are
 * random orthonormal matrices (Haar distributed) and Sigma is a diagonal
 * matrix formed from the user-provided singular values.
 */
template <typename Scalar>
class SigmaMatrixGenerator : public OrthonormalVectorsMatrixGenerator<Scalar> {
  private:
    Eigen::VectorX<Scalar> sigma; ///< The specified singular values.

  public:
    /*!
     * @brief Constructor with a random seed.
     * @param m The number of rows.
     * @param n The number of columns.
     * @param sigma_values A vector of desired singular values. The number of
     *        values must be <= min(m, n).
     */
    SigmaMatrixGenerator(Eigen::Index m, Eigen::Index n,
                         const Eigen::VectorX<Scalar> &sigma_values)
        : OrthonormalVectorsMatrixGenerator<Scalar>(m, n), sigma(sigma_values) {
        assert(sigma.size() <= std::min(m, n) &&
               "Number of singular values cannot exceed the smallest matrix "
               "dimension.");
    }

    /*!
     * @brief Constructor with a specified seed.
     * @param m The number of rows.
     * @param n The number of columns.
     * @param seed The seed for the random number generator.
     * @param sigma_values A vector of desired singular values.
     */
    SigmaMatrixGenerator(Eigen::Index m, Eigen::Index n,
                         std::mt19937::result_type seed,
                         const Eigen::VectorX<Scalar> &sigma_values)
        : OrthonormalVectorsMatrixGenerator<Scalar>(m, n, seed),
          sigma(sigma_values) {
        assert(sigma.size() <= std::min(m, n) &&
               "Number of singular values cannot exceed the smallest matrix "
               "dimension.");
    }

    /*!
     * @brief Gets a string description of the matrix type and its singular
     * values.
     */
    [[nodiscard]] std::string getMatrixType() const override {
        return "matrix with a given set of singular values";
    }

    /*!
     * @brief Generates the matrix using the SVD construction A = U*S*V^T.
     * @return An Eigen::MatrixX<Scalar> of size m x n with the specified sigma.
     *
     * It uses the inherited `generateOrthonormalMatrix` helper to create
     * random U and V matrices, ensuring the random number generator's
     * state is correctly managed.
     */
    [[nodiscard]] Eigen::MatrixX<Scalar> generateMatrix() override {
        auto [m, n] = this->matrixSize;
        Eigen::Index k = sigma.size();

        Eigen::MatrixX<Scalar> U, V;
        {
            // Lock mutex only for RNG access
            std::lock_guard<std::mutex> lock(this->gen_mutex);

            // Generate random orthonormal matrices U (m x k) and V (n x k)
            // using the inherited helper function. This correctly uses the
            // single, stateful random number generator from the base class.
            U = this->generateOrthonormalMatrix(m, k);
            V = this->generateOrthonormalMatrix(n, k);
        } // Mutex unlocked here

        // Construct the final matrix from A = U * Sigma * V^T without holding lock
        return U * sigma.asDiagonal() * V.transpose();
    }
};

} // namespace MatSubset::Bench

#endif // MAT_SUBSET_BENCH_SIGMA_MATRIX_GENERATOR_H