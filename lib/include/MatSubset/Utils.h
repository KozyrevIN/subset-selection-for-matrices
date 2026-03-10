#ifndef MAT_SUBSET_UTILS_H
#define MAT_SUBSET_UTILS_H

#include <Eigen/Core> // For vectors and matrices
#include <Eigen/SVD>  // For Eigen::BDCSVD

#include <cmath>    // For std::sqrt, std::abs, std::fma
#include <iostream> // For std::cerr
#include <limits>   // For std::numeric_limits
#include <tuple>    // For std::tuple

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
     */
    explicit SecularEquationSolver(
        Scalar tolerance = 8 * std::numeric_limits<Scalar>::epsilon())
        : tolerance(tolerance) {}

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
        Eigen::ArrayX<Scalar> v_squared = v.array().square();

        // Deflation
        Eigen::ArrayX<bool> mask =
            Eigen::ArrayX<bool>::Constant(d.size(), false);
        const Scalar mean_v_squared = v_squared.mean();
        Eigen::ArrayX<Scalar> v_squared_deflated(v.size());

        Eigen::Index n = 0;
        Eigen::Index i_1 = 0;
        Eigen::Index i_2 = 0;
        Scalar v_sq_segment_sum = 0;
        while (i_2 < d.size()) {
            v_sq_segment_sum += v_squared(i_2++);
            if ((i_2 == d.size()) ||
                (d(i_2) - d(i_1) >= tolerance * std::abs(d(i_2)))) {
                if (v_sq_segment_sum >=
                    tolerance * (i_2 - i_1) * mean_v_squared) {
                    v_squared_deflated(n++) = v_sq_segment_sum;
                    mask((i_1 + i_2 - 1) / 2) = true;
                }
                i_1 = i_2;
                v_sq_segment_sum = 0;
            }
        }

        v_squared_deflated = v_squared_deflated.head(n).eval();
        Eigen::ArrayX<Scalar> d_deflated(n);
        Eigen::ArrayX<Scalar> d_remaining(d.size() - n);
        i_1 = 0;
        i_2 = 0;
        for (Eigen::Index i = 0; i < d.size(); ++i) {
            if (mask(i)) {
                d_deflated(i_1++) = d(i);
            } else {
                d_remaining(i_2++) = d(i);
            }
        }

        // Solve deflated secular equation
        d_deflated = solveDeflated(d_deflated, v_squared_deflated);

        // Merge all eigenvalues together
        Eigen::VectorX<Scalar> d_new(d.size());
        i_1 = 0;
        i_2 = 0;
        for (Eigen::Index i = 0; i < d.size(); ++i) {
            if (i_1 < n && (i_2 >= d_remaining.size() ||
                            d_deflated(i_1) < d_remaining(i_2))) {
                d_new(i) = d_deflated(i_1++);
            } else {
                d_new(i) = d_remaining(i_2++);
            }
        }

        return d_new;
    }

    Eigen::ArrayX<Scalar>
    solveDeflated(Eigen::ArrayX<Scalar> &d,
                  const Eigen::VectorX<Scalar> &v_squared) const {

        const Eigen::Index n = d.size();
        Eigen::ArrayX<Scalar> d_new(n);

        // Quick return for n = 1 and n = 2
        if (n == 1) {
            d_new(0) = d(0) + v_squared(0);
            return d_new;
        } else if (n == 2) {
            auto [tau, shift, k_origin, k_neighbour] =
                computeInitialGuessAndShiftD(d, v_squared, 0);
            d_new(0) = tau + shift;
            d_new(1) = d.sum() + shift + v_squared.sum() - tau;
            return d_new;
        }

        // Main loop
        const Scalar eps_m = std::numeric_limits<Scalar>::epsilon();
        const int max_iter = 30;

        Scalar increase_sum = 0;
        for (Eigen::Index k = 0; k < n - 1; ++k) {
            bool use_fixed_weight = true;
            auto [tau, shift, k_origin, k_neighbour] =
                computeInitialGuessAndShiftD(d, v_squared, k);
            auto fc = computeFAndComponents(tau, d, v_squared, k);

            for (int i = 0; i < max_iter; ++i) {
                // Convergence check
                if (std::abs(fc.f) <= eps_m * fc.e + eps_m * std::abs(tau) *
                                                         std::abs(fc.f_prime)) {
                    break;
                }

                // Perform iteration
                if (use_fixed_weight) {
                    applyFixedWeightCorrection(tau, d, v_squared, fc, k_origin,
                                               k_neighbour);
                } else {
                    applyMiddleWayCorrection(tau, d, v_squared, fc, k);
                }

                // Compute f at new point
                const Scalar f_prev = fc.f;
                fc = computeFAndComponents(tau, d, v_squared, k);

                // Method switch logic
                if (fc.f * f_prev > static_cast<Scalar>(0) &&
                    std::abs(fc.f) >
                        static_cast<Scalar>(0.1) * std::abs(f_prev)) {
                    use_fixed_weight = !use_fixed_weight;
                }
            }

            d_new(k) = tau + shift;
            increase_sum += tau - d(k);
            d += shift;
        }

        // Handle last eigenvalue (k = n - 1)
        {
            const Eigen::Index k = n - 1;
            const Scalar shift = d(k);
            d -= shift;
            Scalar tau = v_squared.sum() - increase_sum;

            for (int i = 0; i < max_iter; ++i) {
                // Compute f at a new point
                auto fc = computeFAndComponents(tau, d, v_squared, k);

                // Convergence check
                if (std::abs(fc.f) <= eps_m * fc.e + eps_m * std::abs(tau) *
                                                         std::abs(fc.f_prime)) {
                    break;
                }

                // Perform iteration
                const Scalar delta_origin = -tau;
                const Scalar delta_neighbour = d(k - 1) - tau;
                const Scalar delta_prod = delta_origin * delta_neighbour;

                const Scalar a = (delta_origin + delta_neighbour) * fc.f -
                                 delta_prod * fc.f_prime;
                const Scalar b = delta_prod * fc.f;
                const Scalar c = fc.f - delta_neighbour * fc.psi_prime -
                                 v_squared(k) / delta_origin;
                const Scalar disc = std::sqrt(std::fma(a, a, -4 * b * c));

                if (a >= 0) {
                    tau += (a + disc) / (2 * c);
                } else {
                    tau += (2 * b) / (a - disc);
                }
            }

            d_new(k) = tau + shift;
            d += shift;
        }

        return d_new;
    }

  private:
    // Compute psi: sum of v_squared(j) / (d(j) - tau) for j < k
    // Performed in forward order for numerical stability
    Scalar computePsi(Scalar tau, const Eigen::ArrayX<Scalar> &d,
                      const Eigen::ArrayX<Scalar> &v_squared,
                      Eigen::Index k) const {
        Scalar psi = static_cast<Scalar>(0);
        for (Eigen::Index j = 0; j < k; ++j) {
            Scalar delta = d(j) - tau;
            psi += v_squared(j) / delta;
        }
        return psi;
    }

    // Compute phi: sum of v_squared(j) / (d(j) - tau) for j > k
    // Performed in reverse order for numerical stability
    Scalar computePhi(Scalar tau, const Eigen::ArrayX<Scalar> &d,
                      const Eigen::ArrayX<Scalar> &v_squared,
                      Eigen::Index k) const {
        Scalar phi = static_cast<Scalar>(0);
        for (Eigen::Index j = d.size() - 1; j > k; --j) {
            Scalar delta = d(j) - tau;
            phi += v_squared(j) / delta;
        }
        return phi;
    }

    struct FComponents {
        Scalar f;         ///< secular function value
        Scalar f_prime;   ///< derivative
        Scalar psi;       ///< sum_{j<k} v_sq(j) / (d(j) - tau)
        Scalar psi_prime; ///< sum_{j<k} v_sq(j) / (d(j) - tau)^2
        Scalar phi;       ///< sum_{j>k+1}  v_sq(j) / (d(j) - tau)
        Scalar phi_prime; ///< sum_{j>k+1}  v_sq(j) / (d(j) - tau)^2
        Scalar e;         ///< constant for converge
    };

    FComponents computeFAndComponents(Scalar tau,
                                      const Eigen::ArrayX<Scalar> &d,
                                      const Eigen::ArrayX<Scalar> &v_squared,
                                      Eigen::Index k) const {
        Scalar psi = static_cast<Scalar>(0);
        Scalar psi_prime = static_cast<Scalar>(0);
        Scalar phi = static_cast<Scalar>(0);
        Scalar phi_prime = static_cast<Scalar>(0);
        Scalar e_1 = static_cast<Scalar>(0);
        Scalar e_2 = static_cast<Scalar>(0);

        // Compute psi (sum j < k), forward order (j=0 to k - 1)
        // d(0) - tau has largest magnitude, d(k) - tau = -tau
        for (Eigen::Index j = 0; j < k; ++j) {
            Scalar delta = d(j) - tau;
            Scalar term = v_squared(j) / delta;
            psi += term;
            psi_prime += term / delta;
            e_1 += std::abs(psi);
        }
        e_1 += static_cast<Scalar>(5) * std::abs(psi);

        // Compute phi (sum j > k), reverse order (j=n-1 down to k+2)
        // d(n-1) - tau has largest magnitude
        for (Eigen::Index j = d.size() - 1; j > k + 1; --j) {
            Scalar delta = d(j) - tau;
            Scalar term = v_squared(j) / delta;
            phi += term;
            phi_prime += term / delta;
            e_2 += std::abs(phi);
        }
        e_2 += static_cast<Scalar>(5) * std::abs(phi);

        // Boundary term at k (the origin pole, = -tau in the shifted frame)
        const Scalar delta_k = d(k) - tau;
        const Scalar term_k = v_squared(k) / delta_k;
        const Scalar term_k_p = term_k / delta_k;

        // Boundary term at k+1 (the neighbour pole); absent when k = n-1
        Scalar term_k1 = static_cast<Scalar>(0),
               term_k1_p = static_cast<Scalar>(0);
        if (k + 1 < d.size()) {
            const Scalar delta_k1 = d(k + 1) - tau;
            term_k1 = v_squared(k + 1) / delta_k1;
            term_k1_p = term_k1 / delta_k1;
        }

        const Scalar f = static_cast<Scalar>(1) + psi + term_k + term_k1 + phi;
        const Scalar f_prime = psi_prime + term_k_p + term_k1_p + phi_prime;
        const Scalar e =
            static_cast<Scalar>(2) + e_1 + e_2 +
            static_cast<Scalar>(6) * (std::abs(term_k) + std::abs(term_k1)) +
            std::abs(f);

        return {f, f_prime, psi, psi_prime, phi, phi_prime, e};
    }

    // Not for k = n - 1!
    std::tuple<Scalar, Scalar, Eigen::Index, Eigen::Index>
    computeInitialGuessAndShiftD(Eigen::ArrayX<Scalar> &d,
                                 const Eigen::ArrayX<Scalar> &v_squared,
                                 Eigen::Index k) const {

        const Scalar mid = (d(k) + d(k + 1)) * static_cast<Scalar>(0.5);
        const Scalar g_mid = static_cast<Scalar>(1) +
                             computePsi(mid, d, v_squared, k) +
                             computePhi(mid, d, v_squared, k + 1);
        const Scalar h_mid =
            v_squared(k) / (d(k) - mid) + v_squared(k + 1) / (d(k + 1) - mid);

        Scalar tau, shift;
        Eigen::Index k_origin, k_neighbour;
        if (g_mid + h_mid > 0) { // origin at k
            k_origin = k;
            k_neighbour = k + 1;
        } else { // origin at k + 1
            k_origin = k + 1;
            k_neighbour = k;
        }

        shift = d(k_origin);
        d -= shift;
        tau = solveForTwoPoles(g_mid, v_squared(k_origin),
                               v_squared(k_neighbour), d(k_neighbour));

        return {tau, shift, k_origin, k_neighbour};
    }

    Scalar solveForTwoPoles(Scalar c, Scalar v_sq_origin, Scalar v_sq_neighbour,
                            Scalar d_neighbour) const {

        const Scalar a = c * d_neighbour + v_sq_origin + v_sq_neighbour;
        const Scalar b = d_neighbour * v_sq_origin;
        const Scalar disc = std::sqrt(std::fma(a, a, -4 * b * c));

        Scalar tau;
        if (a <= 0) {
            tau = (a - disc) / (2 * c);
        } else {
            tau = (2 * b) / (a + disc);
        }

        return tau;
    }

    // Not for k = n - 1!
    void applyFixedWeightCorrection(Scalar &tau,
                                    const Eigen::ArrayX<Scalar> &d_shifted,
                                    const Eigen::ArrayX<Scalar> &v_squared,
                                    const FComponents &fc,
                                    Eigen::Index k_origin,
                                    Eigen::Index k_neighbour) const {

        const Scalar delta_origin = -tau;
        const Scalar delta_neighbour = d_shifted(k_neighbour) - tau;
        const Scalar delta_sq_neighbour = delta_neighbour * delta_neighbour;

        const Scalar v_sq_origin = v_squared(k_origin);
        const Scalar v_sq_neighbour =
            v_squared(k_neighbour) +
            delta_sq_neighbour * (fc.psi_prime + fc.phi_prime);
        const Scalar c = fc.f - v_sq_origin / delta_origin -
                         v_sq_neighbour / delta_neighbour;

        tau = solveForTwoPoles(c, v_sq_origin, v_sq_neighbour,
                               d_shifted(k_neighbour));
    }

    // Not for k = n - 1!
    void applyMiddleWayCorrection(Scalar &tau,
                                  const Eigen::ArrayX<Scalar> &d_shifted,
                                  const Eigen::ArrayX<Scalar> &v_squared,
                                  const FComponents &fc, Eigen::Index k) const {

        const Scalar delta_1 = tau - d_shifted(k);
        const Scalar delta_2 = tau - d_shifted(k + 1);
        const Scalar delta_prod = delta_1 * delta_2;

        const Scalar a = (delta_1 + delta_2) * fc.f - delta_prod * fc.f_prime;
        const Scalar b = delta_prod * fc.f;
        const Scalar c = fc.f - delta_1 * fc.psi_prime -
                         v_squared(k) / delta_1 - v_squared(k + 1) / delta_2 -
                         delta_2 * fc.phi_prime;
        const Scalar disc = std::sqrt(std::fma(a, a, -4 * b * c));

        if (a <= 0) {
            tau += (a - disc) / (2 * c);
        } else {
            tau += (2 * b) / (a + disc);
        }
    }

    /*!
     * @brief Solves for a root using bisection method.
     * @param d Diagonal elements (shifted).
     * @param v Rank-1 update vector.
     * @param d Shifted diagonal.
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
                std::cerr << "[Bisection] converged in " << iter + 1
                          << " iterations\n";
                return mid;
            }

            // Update bracket
            if (f_mid > static_cast<Scalar>(0)) {
                b = mid;
            } else {
                a = mid;
            }
        }

        std::cerr << "[Bisection] reached max iterations (" << max_iterations
                  << ")\n";
        return (a + b) / static_cast<Scalar>(2);
    }

    Scalar tolerance;   ///< Convergence tolerance
    int max_iterations; ///< Maximum iterations for solver
};

} // namespace MatSubset::Utils

#endif // MAT_SUBSET_UTILS_H