#ifndef MAT_SUBSET_DERANDOMIZED_VOLUME_SELECTOR_H
#define MAT_SUBSET_DERANDOMIZED_VOLUME_SELECTOR_H

#include <cassert>
#include <cmath>    // For std::log10
#include <iostream> // For std::cerr
#include <limits>   // For std::numeric_limits

#include <Eigen/Eigenvalues> // For Eigen::SelfAdjointEigenSolver
#include <Eigen/QR>          // For Eigen::HouseholderQR

#include "SelectorBase.h" // Base class

namespace MatSubset {

/*!
 * @brief Approximates subset selection problem for matrices by derandomizing
 * forward volume sampling.
 * @tparam Scalar The underlying scalar type (e.g., `float`, `double`).
 *
 * Implements derandomized forward volume sampling algorithm. The algorithm
 * greedily selects columns by maintaining an eigendecomposition and computing
 * characteristic polynomials.
 *
 * @note Mild numerical sensitivity: column scores are derived from monomial-
 * basis coefficients of a polynomial whose roots span many orders of magnitude
 * (~`1` to `1/tolerance`). When two candidate columns produce near-tied
 * scores, floating-point rounding determines the choice. Empirically this
 * leads to selections that differ by at most a few columns out of \f$ k \f$
 * across minor algorithm rearrangements, with the resulting
 * pseudoinverse Frobenius norm differing by under ~2% for m <= 200.
 */
template <typename Scalar>
class DerandomizedVolumeSelector : public SelectorBase<Scalar> {
  public:
    /*!
     * @brief Constructor for `DerandomizedVolumeSelector`.
     * @param tolerance Small tolerance for numerical stability. Default:
     * sqrt(machine epsilon).
     */
    explicit DerandomizedVolumeSelector(
        Scalar tolerance = std::sqrt(std::numeric_limits<Scalar>::epsilon()))
        : tolerance(tolerance) {}

    /*!
     * @brief Gets the human-readable name of the algorithm.
     * @return The string "derandomized volume".
     */
    std::string getAlgorithmName() const override {
        return "derandomized volume";
    }

  protected:
    /*!
     * @brief Core implementation for selecting a subset of \f$ k \f$ columns.
     * @param X The input matrix (dimensions \f$ m \times n \f$) from which
     * columns are to be selected. It is assumed that \f$ X \f$ is full rank
     * for theoretical guarantees.
     * @param k The number of columns to select.
     * @return A `std::vector` of `Eigen::Index` containing the 0-based indices
     * of the selected columns.
     */
    std::vector<Eigen::Index>
    selectSubsetImpl(const Eigen::MatrixX<Scalar> &X, Eigen::Index k,
                     Eigen::Index *swap_count) override {
        // Initialization
        Eigen::Index m = X.rows();
        Eigen::Index n = X.cols();

        Eigen::HouseholderQR<Eigen::MatrixX<Scalar>> qr(X.transpose());
        Eigen::MatrixX<Scalar> Q =
            (qr.householderQ() * Eigen::MatrixX<Scalar>::Identity(n, m))
                .transpose();

        std::vector<Eigen::Index> selected_indices;
        std::vector<Eigen::Index> remaining_indices(n);
        for (Eigen::Index j = 0; j < n; ++j) {
            remaining_indices[static_cast<size_t>(j)] = j;
        }

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<Scalar>> eigensolver;
        // Eigenvalues are sorted in non-decreasing order
        Eigen::VectorX<Scalar> lambda = Eigen::VectorX<Scalar>::Zero(m);
        Eigen::Index r = 0;

        // Main loop
        for (Eigen::Index t = 0; t < k; ++t) {
            // Deflation
            const bool r_less_m = (r < m);
            Eigen::Index x_minus_1_deg = n - m - t - 1;
            assert(x_minus_1_deg >= -1 &&
                   "x_minus_1_deg can't be less than -1");

            auto [start, len] = deflateAndCheckOverflow(lambda, x_minus_1_deg);
            Eigen::VectorX<Scalar> lambda_deflated = lambda.segment(start, len);
            Eigen::MatrixX<Scalar> Q_deflated(len + r_less_m, n - t);
            Q_deflated.bottomRows(len) = Q.middleRows(start, len);
            if (r_less_m) {
                Q_deflated.row(0) = Q.topRows(start).colwise().norm();
            }

            // Target degrees
            const Eigen::Index min_deg_p = n - k - 1 - r_less_m;
            const Eigen::Index max_deg_p = n - k;

            const Eigen::Index min_deg_g = n - k - 1;
            const Eigen::Index max_deg_g = n - k;

            // Construction of auxiliary polynomials
            Eigen::VectorX<Scalar> lambda_inv = lambda_deflated.cwiseInverse();
            Eigen::VectorX<Scalar> p_0 = buildPolynomialFromRoots(lambda_inv);
            Eigen::MatrixX<Scalar> g = buildQuotentPolynomials(p_0, lambda_inv);

            // Computation of characteristic polynomial coefficients
            Eigen::MatrixX<Scalar> p(2, n - t);
            if (x_minus_1_deg >= 0) {
                p_0 = applyBinomials(p_0, x_minus_1_deg, min_deg_p, max_deg_p);
                g = applyBinomials(g, x_minus_1_deg, min_deg_g, max_deg_g);

                Eigen::MatrixX<Scalar> Lambda_inv_Q_squared =
                    lambda_inv.asDiagonal() *
                    Q_deflated.bottomRows(len).cwiseAbs2();
                Eigen::RowVectorX<Scalar> p_factor =
                    1 + Lambda_inv_Q_squared.colwise().sum().array();
                Eigen::RowVectorX<Scalar> p_x_factor =
                    Eigen::RowVectorX<Scalar>::Zero(n - t);
                if (r_less_m) {
                    p_x_factor = -Q_deflated.row(0).cwiseAbs2();
                }
                Eigen::MatrixX<Scalar> g_factor =
                    lambda_inv.asDiagonal() * Lambda_inv_Q_squared;

                p = p_0.tail(2) * p_factor + p_0.head(2) * p_x_factor +
                    g * g_factor;
            } else {
                p_0 = p_0.segment(min_deg_g, 2); // min_deg_g is intentional
                g = g.middleRows(min_deg_g, 2);

                Eigen::RowVectorX<Scalar> p_factor =
                    -Q_deflated.row(0).cwiseAbs2();
                Eigen::MatrixX<Scalar> g_factor =
                    (lambda_inv.array() / (1 - lambda_deflated.array()))
                        .matrix()
                        .asDiagonal() *
                    Q_deflated.bottomRows(len).cwiseAbs2();

                p = p_0 * p_factor + g * g_factor;
            }

            // Array with final |c_{n - k - 1} / c_{n - k}|
            Eigen::ArrayX<Scalar> ratios =
                (p.row(0).array() / p.row(1).array()).abs();
            Eigen::Index s;
            ratios.minCoeff(&s);

            Eigen::VectorX<Scalar> q_s = Q.col(s);
            selected_indices.push_back(
                remaining_indices[static_cast<size_t>(s)]);
            if (static_cast<Eigen::Index>(remaining_indices.size()) - 1 != s) {
                remaining_indices[s] = remaining_indices.back();
                Q.col(s) = Q.col(Q.cols() - 1);
            }
            remaining_indices.pop_back();
            Q.conservativeResize(Eigen::NoChange, Q.cols() - 1);

            Eigen::MatrixX<Scalar> M = q_s * q_s.transpose();
            M.diagonal() += lambda;
            eigensolver.compute(M);
            lambda = eigensolver.eigenvalues();
            Q = eigensolver.eigenvectors().transpose() * Q;

            // Update r to count nonzero eigenvalues (within tolerance)
            r = (lambda.array() > tolerance).count();
        }

        return selected_indices;
    }

  private:
    /*!
     * @brief Numerical tolerance for checking \f$ w_j > 0 \f$.
     */
    Scalar tolerance;

    /*!
     * @brief Build polynomial \f$ p(x) = \prod_i (x - \text{root}_i) \f$ via
     * incremental left-to-right multiplication, consuming roots in
     * ascending-magnitude order.
     * @param roots Vector of roots, **assumed sorted in descending order**
     * (which is what the caller produces by inverting the ascending
     * `lambda_deflated`).
     * @return Polynomial coefficients in standard form (size = deg + 1).
     *
     * Each step multiplies the running polynomial by \f$(x - r_i)\f$, so
     * coefficient magnitudes grow as roots are absorbed. Consuming the
     * smallest roots first keeps the intermediate polynomial well-scaled
     * for as long as possible, deferring catastrophic cancellation toward
     * the end where fewer steps remain to amplify error. (Equivalent in
     * exact arithmetic to any other ordering; rounds better in finite
     * precision when \f$ |\text{roots}| \f$ varies over many orders of
     * magnitude.)
     */
    Eigen::VectorX<Scalar>
    buildPolynomialFromRoots(const Eigen::VectorX<Scalar> &roots) const {
        const Eigen::Index deg = roots.size();
        Eigen::VectorX<Scalar> p = Eigen::VectorX<Scalar>::Zero(deg + 1);
        p(deg) = static_cast<Scalar>(1);

        // Consume roots smallest-magnitude first: roots(deg - i) walks the
        // input in reverse (the caller passes a descending vector).
        for (Eigen::Index i = 1; i <= deg; ++i) {
            p.segment(deg - i, i) -= roots(deg - i) * p.tail(i).eval();
        }

        return p;
    }

    /*!
     * @brief Deflate eigenvalues equal to 0 or 1 and check for overflow.
     * @param lambda Vector of eigenvalues sorted in increasing order.
     * @param x_minus_1_deg Reference to exponent of (x-1), incremented by count
     * of eigenvalues equal to 1.
     * @return Pair (start, len) where start is index of first active eigenvalue
     * and len is count of active eigenvalues.
     */
    std::pair<Eigen::Index, Eigen::Index>
    deflateAndCheckOverflow(const Eigen::VectorX<Scalar> &lambda,
                            Eigen::Index &x_minus_1_deg) const {
        Eigen::Index start = 0;

        // Skip eigenvalues equal to 0 (within tolerance)
        while (start < lambda.size() && lambda[start] < tolerance) {
            start++;
        }

        // Find where eigenvalues equal to 1 begin
        Eigen::Index ones_start = start;
        while (ones_start < lambda.size() &&
               lambda[ones_start] < Scalar(1) - tolerance) {
            ones_start++;
        }
        x_minus_1_deg += lambda.size() - ones_start;

        // Return segment containing only eigenvalues in (0, 1)
        const Eigen::Index len = ones_start - start;

        // Check for potential overflow on the active eigenvalues
        const Scalar max_log_val =
            std::log10(std::numeric_limits<Scalar>::max());
        const Scalar safety_buffer =
            static_cast<Scalar>(len + 1) / 3 + std::log10(tolerance);
        const Scalar safe_limit = max_log_val - safety_buffer;

        Scalar log_magnitude = 0;
        for (Eigen::Index i = start; i < start + len; ++i) {
            log_magnitude += std::log10(1 / lambda[i]);
        }

        if (log_magnitude > safe_limit) {
            std::cerr << "Warning: Predicted overflow! Log magnitude sum: "
                      << log_magnitude << std::endl;
        }

        return {start, len};
    }

    /*!
     * @brief Build quotient polynomials \f$ g_i(x) = p_0(x) / (x - r_i) \f$
     * via hybrid (two-way) synthetic division.
     * @param p_0 Coefficients of \f$ p_0(x) = \prod_i (x - r_i) \f$, standard
     * form (size = num_roots + 1).
     * @param roots Vector of roots \f$ r_i \ge 0 \f$.
     * @return Matrix where column \f$ i \f$ contains coefficients of
     * \f$ g_i(x) \f$ in standard form (size = num_roots).
     *
     * Cost: \f$ O(\text{num\_roots}^2) \f$ instead of the naive
     * \f$ O(\text{num\_roots}^3) \f$ that would result from building each
     * column independently.
     *
     * Stability strategy: each column is computed by running synthetic
     * division from both ends simultaneously and meeting at the peak.
     * - Forward recurrence (high-to-low) scales errors by \f$ r_i \f$.
     * - Backward recurrence (low-to-high) divides errors by \f$ r_i \f$.
     * At each step we advance whichever front currently has smaller
     * magnitude; this naturally places the meeting point near each
     * column's peak coefficient — keeping relative error bounded.
     */
    Eigen::MatrixX<Scalar>
    buildQuotentPolynomials(const Eigen::VectorX<Scalar> &p_0,
                            const Eigen::VectorX<Scalar> &roots) const {
        const Eigen::Index num_roots = roots.size();
        const Eigen::Index g_deg = num_roots - 1;
        Eigen::MatrixX<Scalar> g(num_roots, num_roots);

        for (Eigen::Index i = 0; i < num_roots; ++i) {
            const Scalar r = roots(i);

            // Front anchors: forward starts at degree g_deg, backward at 0.
            Scalar fwd = p_0(num_roots); // b_{g_deg}
            Scalar bwd = -p_0(0) / r;    // b_0
            Eigen::Index hi = g_deg;     // next forward write
            Eigen::Index lo = 0;         // next backward write
            g(hi, i) = fwd;
            g(lo, i) = bwd;

            // Race the two recurrences toward each other. At each step,
            // advance whichever front currently has smaller magnitude;
            // that step pushes it toward the (larger) peak — keeping
            // relative error small. They meet at the peak.
            while (hi - lo > 1) {
                if (std::abs(fwd) < std::abs(bwd)) {
                    // Forward step: b_{hi-1} = c_hi + r * b_hi
                    fwd = p_0(hi) + r * fwd;
                    --hi;
                    g(hi, i) = fwd;
                } else {
                    // Backward step: b_{lo+1} = (b_lo - c_{lo+1}) / r
                    bwd = (bwd - p_0(lo + 1)) / r;
                    ++lo;
                    g(lo, i) = bwd;
                }
            }
        }

        return g;
    }

    /*!
     * @brief Multiply polynomial matrix by \f$ (x-1)^{\text{x\_minus\_1\_deg}}
     * \f$ up to a constant, keeping only coefficients for degrees [min_deg,
     * max_deg].
     * @param poly Polynomial matrix where each column is a polynomial.
     * @param x_minus_1_deg Exponent of (x-1).
     * @param min_deg Lowest degree to keep.
     * @param max_deg Highest degree to keep.
     * @return New matrix with trimmed polynomial coefficients.
     */
    Eigen::MatrixX<Scalar> applyBinomials(const Eigen::MatrixX<Scalar> &poly,
                                          Eigen::Index x_minus_1_deg,
                                          Eigen::Index min_deg,
                                          Eigen::Index max_deg) const {

        const Eigen::Index input_deg = poly.rows() - 1;

        // Binomial indices we need
        const Eigen::Index min_binom_deg =
            std::max(min_deg - input_deg, static_cast<Eigen::Index>(0));
        const Eigen::Index max_binom_deg = std::min(max_deg, x_minus_1_deg);
        const Eigen::Index num_coeffs = max_binom_deg - min_binom_deg + 1;

        Eigen::MatrixX<Scalar> poly_new =
            Eigen::MatrixX<Scalar>::Zero(max_deg - min_deg + 1, poly.cols());
        // No valid coefficients: poly is empty, or the requested output
        // degree range lies entirely outside [0, input_deg + x_minus_1_deg]
        if (poly.rows() == 0 || num_coeffs <= 0) {
            return poly_new;
        }

        // We sort binomials from one corresponding to largest deg to smallest
        Eigen::VectorX<Scalar> binoms(num_coeffs);
        binoms(0) = 1;
        for (Eigen::Index i = 1; i < num_coeffs; ++i) {
            const Scalar idx = static_cast<Scalar>(max_binom_deg - i);
            binoms(i) = -binoms(i - 1) * (idx + 1) / (x_minus_1_deg - idx);
        }

        // Apply to poly (column-wise)
        for (Eigen::Index i = min_deg; i <= max_deg; ++i) {
            Eigen::Index shift = num_coeffs + min_binom_deg - i - 1;
            // Constraint 1: 0 <= j < p.rows()
            // Constraint 2: 0 <= j + shift < binoms.size()
            Eigen::Index j_start =
                std::max(static_cast<Eigen::Index>(0), -shift);
            Eigen::Index j_end = std::min(input_deg, binoms.size() - shift - 1);
            Eigen::Index len = j_end - j_start + 1;

            if (len >= 1) {
                poly_new.row(i - min_deg) =
                    binoms.segment(j_start + shift, len).transpose() *
                    poly.middleRows(j_start, len);
            }
        }
        return poly_new;
    }
};

} // namespace MatSubset

#endif // MAT_SUBSET_DERANDOMIZED_VOLUME_SELECTOR_H
