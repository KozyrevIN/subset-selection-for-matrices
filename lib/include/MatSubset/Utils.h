#ifndef MAT_SUBSET_UTILS_H
#define MAT_SUBSET_UTILS_H

#include <Eigen/Core> // For vectors and matrices
#include <Eigen/SVD>  // For Eigen::BDCSVD

#include <cmath>  // For std::sqrt
#include <limits> // For std::numeric_limits

#include "Enums.h" // For MatSubset::Norm

namespace MatSubset::Utils {

/**
 * @brief Calculates the specified norm of the Moore-Penrose pseudoinverse of a
 * matrix.
 *
 * This function computes the singular value decomposition (SVD) of the input
 * matrix \f$ X \f$ to analyze its singular values.
 *
 * If the matrix is determined to be rank-deficient (i.e., its smallest singular
 * value is close to zero relative to its largest singular value), the function
 * returns infinity. Otherwise, it calculates the desired norm of the
 * pseudoinverse using the singular values of \f$ X \f$. The singular values of
 * the pseudoinverse are the reciprocals of the non-zero singular values of the
 * original matrix.
 *
 * @tparam Scalar The Scalar type of the matrix elements (e.g., float, double).
 * @tparam norm A compile-time constant of type `MatSubset::Norm` specifying
 * which norm to calculate. Supported values are `MatSubset::Norm::Frobenius`
 * and `MatSubset::Norm::Spectral`.
 * @param X The input matrix for which the pseudoinverse norm is to be
 * calculated.
 * @return The calculated norm (Frobenius or Spectral) of the pseudoinverse of
 * \f$ X \f$. Returns `infinity` if \f$ X \f$ is rank-deficient.
 *
 * @note If an unsupported `MatSubset::Norm` value is provided as a template
 * argument, a static_assert will cause a compile-time error.
 */
template <typename Scalar, Norm norm>
Scalar pinv_norm(const Eigen::MatrixX<Scalar> &X) {

    // Compute the SVD of the input matrix. We only need the singular values.
    Eigen::BDCSVD<Eigen::MatrixX<Scalar>> svd(X);
    const Eigen::Array<Scalar, Eigen::Dynamic, 1> S = svd.singularValues();

    // Handle empty or zero-sized matrices.
    if (S.size() == 0) {
        return static_cast<Scalar>(0);
    }

    // Check for rank deficiency. If the smallest singular value is close to
    // zero compared to the largest, the matrix is ill-conditioned or singular.
    // In this case, the norm of the pseudoinverse is considered infinite.
    const Scalar tolerance = S(0) * std::numeric_limits<Scalar>::epsilon();
    if (S(S.size() - 1) <= tolerance) {
        return std::numeric_limits<Scalar>::infinity();
    }

    if constexpr (norm == Norm::Frobenius) {
        // The Frobenius norm of the pseudoinverse is sqrt(sum(1/s_i^2)),
        // where s_i are the non-zero singular values of the original matrix.
        return std::sqrt(S.inverse().square().sum());

    } else if constexpr (norm == Norm::Spectral) {
        // The spectral norm of the pseudoinverse is 1/s_min, where
        // s_min is the smallest non-zero singular value of the original matrix.
        return static_cast<Scalar>(1) / S(S.size() - 1);

    } else {
        // This structure allows easily adding more norms in the future.
        // For any other enum value, this will cause a compile-time error.
        static_assert(false, "This norm type is unsupported in pinv_norm!");
        // Return a dummy value to satisfy compilers for uninstantiated
        // branches, though static_assert(false,...) should ideally prevent
        // instantiation. Some compilers might still require a return path.
        return Scalar{};
    }
}

} // namespace MatSubset::Utils

#endif // MAT_SUBSET_UTILS_H