#ifndef MAT_SUBSET_UTILS_H
#define MAT_SUBSET_UTILS_H

#include <Eigen/Core> // For vectors and matrices
#include <Eigen/SVD>  // For Eigen::BDCSVD

#include <cmath>  // For std::sqrt, std::abs
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

/*!
 * @brief Solver for secular equations arising from diagonal plus rank-1
 * updates.
 * @tparam Scalar The underlying scalar type (e.g., `float`, `double`).
 *
 * This class computes eigenvalues of matrices of the form:
 * \f$ M = \text{diag}(d) + vv^T \f$
 *
 * where \f$ d \f$ is a vector of diagonal elements and \f$ v \f$ is a column
 * vector. The eigenvalues satisfy the secular equation:
 * \f$ 1 + \sum_{i} \frac{v_i^2}{d_i - \lambda} = 0 \f$
 *
 * This solver uses bisection method for robustness. The approach is
 * significantly faster than full eigendecomposition:
 * - Full eigendecomposition: O(n^3)
 * - Secular equation solver: O(n^2)
 *
 * @note The diagonal elements \f$ d \f$ must be sorted in increasing order for
 * correct behavior.
 * @note This implementation assumes \f$ v_i \neq 0 \f$ for non-deflated
 * entries. Zero entries in \f$ v \f$ should be handled by deflation before
 * calling the solver.
 */
template <typename Scalar> class SecularEquationSolver {
  public:
    /*!
     * @brief Constructor for `SecularEquationSolver`.
     * @param tolerance Convergence tolerance for iterative solver. Default: 8 *
     * machine epsilon.
     * @param max_iter Maximum number of iterations for the solver.
     * Default: 100.
     */
    explicit SecularEquationSolver(
        Scalar tolerance = 8 * std::numeric_limits<Scalar>::epsilon(),
        int max_iter = 100)
        : tolerance(tolerance), max_iterations(max_iter) {}

    /*!
     * @brief Computes all eigenvalues of \f$ \text{diag}(d) + vv^T \f$.
     * @param d Diagonal elements (must be sorted in increasing order).
     * @param v Rank-1 update vector.
     * @return Eigenvalues sorted in increasing order.
     *
     * For each eigenvalue, the solver identifies the appropriate bracketing
     * interval based on interlacing properties of eigenvalues, then applies
     * bisection to find the root.
     */
    Eigen::VectorX<Scalar> solve(const Eigen::VectorX<Scalar> &d,
                                 const Eigen::VectorX<Scalar> &v) const {
        // Deflation type 1: identify non-zero v components
        Eigen::ArrayX<bool> mask =
            (v.array().abs2() > v.array().abs2().mean() * tolerance);

        // Deflation type 2: merge clusters of equal diagonal elements
        Eigen::Index n = 0;
        Eigen::VectorX<Scalar> v_deflated(v.size());
        Eigen::Index i_1 = 0;
        Eigen::Index i_2 = 1;
        while (i_2 <= d.size()) {
            if ((i_2 == d.size()) ||
                (d(i_2) - d(i_1) >= tolerance * std::abs(d(i_2)))) {
                if (mask.segment(i_1, i_2 - i_1).sum() > 0) {
                    v_deflated(n++) = v.segment(i_1, i_2 - i_1).norm();
                    mask.segment(i_1, i_2 - i_1) =
                        Eigen::VectorX<bool>::Constant(i_2 - i_1, false);
                    mask((i_1 + i_2 - 1) / 2) = true;
                }
                i_1 = i_2;
            }
            ++i_2;
        }

        v_deflated = v_deflated.head(n).eval();
        Eigen::VectorX<Scalar> d_deflated(n);
        Eigen::VectorX<Scalar> d_remaining(d.size() - n);
        i_1 = 0;
        i_2 = 0;
        for (Eigen::Index i = 0; i < d.size(); ++i) {
            if (mask(i)) {
                d_deflated(i_1++) = d(i);
            } else {
                d_remaining(i_2++) = d(i);
            }
        }

        // Solve secular equation for each deflated eigenvalue
        Eigen::VectorX<Scalar> d_deflated_new(n);
        for (Eigen::Index i = 0; i < n; ++i) {
            Scalar left = d_deflated(i);
            Scalar right = (i == (n - 1)) ? d_deflated(i) + v.squaredNorm()
                                          : d_deflated(i + 1);
            d_deflated_new(i) =
                solveInInterval(d_deflated, v_deflated, i, left, right);
        }

        // Merge all eigenvalues together
        Eigen::VectorX<Scalar> d_new(d.size());
        i_1 = 0;
        i_2 = 0;
        for (Eigen::Index i = 0; i < d.size(); ++i) {
            if (i_1 < n && (i_2 >= d_remaining.size() ||
                            d_deflated_new(i_1) < d_remaining(i_2))) {
                d_new(i) = d_deflated_new(i_1++);
            } else {
                d_new(i) = d_remaining(i_2++);
            }
        }

        return d_new;
    }

    /*!
     * @brief Computes a single eigenvalue in a given interval.
     * @param d Diagonal elements.
     * @param v Rank-1 update vector.
     * @param index The index for which we are solving for
     * @param left Left endpoint of bracketing interval.
     * @param right Right endpoint of bracketing interval.
     * @return The eigenvalue in the interval [left, right].
     *
     * Uses bisection to find the root of the secular equation in the specified
     * interval. The equation is shifted to improve numerical stability.
     */
    Scalar solveInInterval(const Eigen::VectorX<Scalar> &d,
                           const Eigen::VectorX<Scalar> &v, Eigen::Index index,
                           Scalar left, Scalar right) const {
        // Choose shift to improve numerical stability
        // We shift relative to the midpoint of the interval
        Scalar mid = (left + right) / static_cast<Scalar>(2);
        Scalar f_mid = evaluateSecularEquation(mid, d, v);

        // Choose shift as the endpoint closer to the root
        Scalar shift;
        if (f_mid > static_cast<Scalar>(0)) {
            shift = left;
            left = 0;
            right = mid - shift;
        } else {
            shift = right;
            left = mid - shift;
            right = 0;
        }

        // Create shifted diagonal
        Eigen::VectorX<Scalar> d_shifted = d.array() - shift;

        // Sort diagonal to put smallest absolute values at the end to ensure
        // stable summation
        Eigen::VectorX<Scalar> d_sorted(d.size());
        Eigen::VectorX<Scalar> v_sorted(v.size());
        Eigen::Index neg_ptr = 0;            // moves forward through negatives
        Eigen::Index pos_ptr = d.size() - 1; // moves backward through positives

        for (Eigen::Index i = 0; i < d.size(); ++i) {
            bool take_neg =
                (neg_ptr <= index) &&
                (pos_ptr <= index ||
                 std::abs(d_shifted(neg_ptr)) > std::abs(d_shifted(pos_ptr)));
            if (take_neg) {
                d_sorted(i) = d_shifted(neg_ptr);
                v_sorted(i) = v(neg_ptr);
                ++neg_ptr;
            } else {
                d_sorted(i) = d_shifted(pos_ptr);
                v_sorted(i) = v(pos_ptr);
                --pos_ptr;
            }
        }

        return shift + solveWithBisection(d_sorted, v_sorted, left, right);
    }

  private:
    /*!
     * @brief Evaluates the secular equation at a given point (unshifted).
     * @param lambda The point at which to evaluate.
     * @param d Diagonal elements.
     * @param v Rank-1 update vector.
     * @return Value of \f$ 1 + \sum_{i} \frac{v_i^2}{d_i - \lambda} \f$
     */
    Scalar evaluateSecularEquation(Scalar lambda,
                                   const Eigen::VectorX<Scalar> &d,
                                   const Eigen::VectorX<Scalar> &v) const {
        return 1 + (v.cwiseAbs2().array() / (d.array() - lambda)).sum();
    }

    /*!
     * @brief Solves for a root using bisection method.
     * @param d Diagonal elements (original).
     * @param v Rank-1 update vector.
     * @param d_shifted Shifted diagonal.
     * @param shift The shift value.
     * @param left Left endpoint in shifted coordinates.
     * @param right Right endpoint in shifted coordinates.
     * @return The computed eigenvalue (in original, unshifted coordinates).
     *
     * Robust method that guarantees convergence when the function has
     * opposite signs at the endpoints.
     */
    Scalar solveWithBisection(const Eigen::VectorX<Scalar> &d,
                              const Eigen::VectorX<Scalar> &v, Scalar left,
                              Scalar right) const {

        Scalar a = left;
        Scalar b = right;

        for (int iter = 0; iter < max_iterations; ++iter) {
            Scalar mid = (a + b) / static_cast<Scalar>(2);
            Scalar f_mid = evaluateSecularEquation(mid, d, v);

            // Check convergence
            if (std::abs(b - a) < tolerance * (right - left)) {
                return mid;
            }

            // Update bracket
            if (f_mid > static_cast<Scalar>(0)) {
                b = mid;
            } else {
                a = mid;
            }
        }

        return (a + b) / static_cast<Scalar>(2);
    }

    Scalar tolerance;   ///< Convergence tolerance
    int max_iterations; ///< Maximum iterations for solver
};

} // namespace MatSubset::Utils

#endif // MAT_SUBSET_UTILS_H