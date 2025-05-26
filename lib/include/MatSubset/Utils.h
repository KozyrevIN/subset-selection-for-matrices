#ifndef MAT_SUBSET_UTILS_H
#define MAT_SUBSET_UTILS_H

#include <Eigen/Core>
#include <Eigen/SVD> // For Eigen::JacobiSVD and methods like .pseudoInverse() often rely on SVD concepts
#include <Eigen/QR>  // For Eigen::CompleteOrthogonalDecomposition

#include "Enums.h" // For MatSubset::Norm

namespace MatSubset {
namespace Utils {

/**
 * @brief Calculates the specified norm of the Moore-Penrose pseudoinverse of a matrix.
 *
 * This function first computes the Moore-Penrose pseudoinverse of the input matrix X
 * using Eigen's Complete Orthogonal Decomposition. Then, it calculates either the
 * Frobenius norm or the Spectral norm of this pseudoinverse.
 *
 * @tparam Scalar The Scalar type of the matrix elements (e.g., float, double).
 * @tparam norm A compile-time constant of type MatSubset::Norm specifying which norm to calculate.
 *              Supported values are MatSubset::Norm::Frobenius and MatSubset::Norm::Spectral.
 * @param X The input matrix for which the pseudoinverse norm is to be calculated.
 * @return The calculated norm (Frobenius or Spectral) of the pseudoinverse of X.
 *
 * @note If an unsupported MatSubset::Norm value is provided as a template argument,
 *       a static_assert will cause a compile-time error.
 * @throws This function does not throw exceptions directly, but Eigen functions might
 *         if memory allocation fails.
 *
 * Example usage:
 * @code
 *   Eigen::MatrixXd A(3, 2);
 *   A << 1, 2,
 *        3, 4,
 *        5, 6;
 *   double frobenius_norm_pinv_A = MatSubset::Utils::pinv_norm<double, MatSubset::Norm::Frobenius>(A);
 *   double spectral_norm_pinv_A = MatSubset::Utils::pinv_norm<double, MatSubset::Norm::Spectral>(A);
 * @endcode
 */
template <typename Scalar, MatSubset::Norm norm_type>
Scalar pinv_norm(const Eigen::MatrixX<Scalar>& X) {

    if constexpr (norm_type == MatSubset::Norm::Frobenius) {
        // Compute pseudo-inverse using Complete Orthogonal Decomposition (robust)
        // and then its Frobenius norm.
        return X.completeOrthogonalDecomposition().pseudoInverse().norm();

    } else if constexpr (norm_type == MatSubset::Norm::Spectral) {
        // Compute pseudo-inverse
        auto pinv = X.completeOrthogonalDecomposition().pseudoInverse();
        // The spectral norm is the largest singular value.
        // We need to compute SVD of the pseudo-inverse matrix.
        Eigen::JacobiSVD<Eigen::MatrixX<Scalar>> svd_of_pinv(pinv, Eigen::ComputeThinU | Eigen::ComputeThinV); // Eigen::SingularValuesOnly would also work if available and desired
        if (svd_of_pinv.singularValues().size() == 0) {
            // This case can happen if pinv is a zero matrix (e.g. pseudoinverse of a zero matrix)
            // or has zero dimensions, though Eigen usually handles dimensions.
            // The largest singular value of a zero matrix is 0.
            return static_cast<Scalar>(0);
        }
        return svd_of_pinv.singularValues()(0); // singularValues() are sorted in descending order

    } else {
        // This structure allows easily adding more norms in the future.
        // For any other enum value, this will cause a compile-time error.
        static_assert(false, "This norm type is unsupported in pinv_norm!");
        // Return a dummy value to satisfy compilers for uninstantiated branches,
        // though static_assert(false,...) should ideally prevent instantiation.
        // Some compilers might still require a return path.
        return Scalar{};
    }
}

} // namespace Utils
} // namespace MatSubset

#endif // MAT_SUBSET_UTILS_H