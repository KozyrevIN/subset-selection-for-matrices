#ifndef MAT_SUBSET_DERANDOMIZED_VOLUME_SELECTOR_H
#define MAT_SUBSET_DERANDOMIZED_VOLUME_SELECTOR_H

#include <cassert>
#include <cmath>    // For std::log10
#include <iostream> // For std::cerr
#include <limits>   // For std::numeric_limits

#include <Eigen/Eigenvalues> // For Eigen::SelfAdjointEigenSolver
#include <Eigen/QR>          // For Eigen::HouseholderQR

#include "SelectorBase.h" // Base class
#include "Utils.h"        // For SecularEquationSolver

namespace MatSubset {

/*!
 * @brief Approximates subset selection problem for matrices by derandomizing
 * forward volume sampling.
 * @tparam Scalar The underlying scalar type (e.g., `float`, `double`).
 *
 * Implements derandomized forward volume sampling algorithm. The algorithm
 * greedily selects columns by maintaining an eigendecomposition and computing
 * characteristic polynomials.
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
    std::vector<Eigen::Index> selectSubsetImpl(const Eigen::MatrixX<Scalar> &X,
                                               Eigen::Index k) override {
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
        // Eigenvvalues are sorted in non-decreasing order
        Eigen::VectorX<Scalar> lambda = Eigen::VectorX<Scalar>::Zero(m);
        Eigen::Index r = 0;

        // Main loop
        for (Eigen::Index t = 0; t < k; ++t) {
            // Deflation
            const bool r_less_m = (r < m);
            Eigen::Index x_minus_1_deg = n - m - t - 1;
            auto [start, len] = deflateAndCheckOverflow(lambda, x_minus_1_deg);
            Eigen::VectorX<Scalar> lambda_deflated = lambda.segment(start, len);
            Eigen::MatrixX<Scalar> Q_deflated(len + r_less_m, n - t);
            Q_deflated.bottomRows(len) = Q.middleRows(start, len);
            if (r_less_m) {
                Q_deflated.row(0) = Q.topRows(start).colwise().norm();
            }

            // Target degrees
            const Eigen::Index min_deg = n - k - 1;
            const Eigen::Index max_deg = n - k;
            Eigen::MatrixX<Scalar> p;

            // Construction of characteristic polynomials
            if (r < m) { // Phase 1 (r < m)
                p = Eigen::MatrixX<Scalar>::Zero(len + 2, n - t);

                for (Eigen::Index j = 0; j < n - t; ++j) {
                    Eigen::VectorX<Scalar> lambda_j =
                        updateEigenvalues(lambda_deflated, Q_deflated.col(j));

                    // Additional deflation
                    if (lambda_j(0) < tolerance) {
                        lambda_j = lambda_j.tail(lambda_j.size() - 1);
                    }
                    if (x_minus_1_deg < 0) {
                        assert(x_minus_1_deg == -1 &&
                               "x_minus_1_deg can't be less than -1");
                        x_minus_1_deg = 0;
                        lambda_j = lambda_j.head(lambda_j.size() - 1);
                    }

                    p.col(j).head(lambda_j.size() + 1) =
                        buildPolynomialFromRoots(lambda_j.cwiseInverse());
                }

                // Multiply by (x - 1) in required degree
                p = applyBinomials(p, x_minus_1_deg, min_deg, max_deg);

            } else { // Phase 2 (r = m)
                Eigen::VectorX<Scalar> lambda_inv =
                    lambda_deflated.cwiseInverse();
                Eigen::VectorX<Scalar> p_0 =
                    buildPolynomialFromRoots(lambda_inv);
                Eigen::MatrixX<Scalar> g = buildQuotentPolynomials(lambda_inv);

                // We'll later need to divide by (x - 1)
                if (x_minus_1_deg >= 0) {
                    p_0 = applyBinomials(p_0, x_minus_1_deg, min_deg, max_deg);
                    g = applyBinomials(g, x_minus_1_deg, min_deg, max_deg);
                }

                // Matrix-determinant lemma essentially
                Eigen::MatrixX<Scalar> Lambda_inv_Q_squared =
                    lambda_inv.asDiagonal() * Q_deflated.cwiseAbs2();
                Eigen::ArrayX<Scalar> scale_coeffs =
                    1 + Lambda_inv_Q_squared.colwise().sum().array();
                Eigen::ArrayXX<Scalar> g_factor =
                    (g * lambda_inv.asDiagonal() * Lambda_inv_Q_squared)
                        .array();
                p = p_0.replicate(1, n - t);
                p.topRows(g.rows()) +=
                    (g_factor.rowwise() / scale_coeffs.transpose()).matrix();

                // Divide by (x - 1) if needed
                if (x_minus_1_deg < 0) {
                    assert(x_minus_1_deg == -1 &&
                           "x_minus_1_deg can't be less than -1");
                    p = dividePolynomialByLinear(p, 1);
                    p = p.middleRows(min_deg, 2).eval();
                }
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
     * @brief Divide polynomials by (x - root) using synthetic division.
     * @param poly Polynomial coefficients matrix where each column is a
     * polynomial [c_0, c_1, ..., c_n] with poly(x) = c_0 + c_1*x + ... +
     * c_n*x^n.
     * @param root The root value to divide by.
     * @return Quotient polynomials (degree reduced by 1 for each column).
     *
     * Assumes poly(root) = 0 or is very small (valid division).
     */
    Eigen::MatrixX<Scalar>
    dividePolynomialByLinear(const Eigen::MatrixX<Scalar> &poly,
                             Scalar root) const {
        const Eigen::Index deg = poly.rows() - 1;
        Eigen::MatrixX<Scalar> quotient(deg, poly.cols());

        // Synthetic division: work from highest degree down for each column
        quotient.row(deg - 1) = poly.row(deg);
        for (Eigen::Index i = deg - 2; i >= 0; --i) {
            quotient.row(i) = poly.row(i + 1) + root * quotient.row(i + 1);
        }

        return quotient;
    }

    /*!
     * @brief Build polynomial \f$ p(x) = \prod_{i} (x - \text{root}_i) \f$
     * from roots.
     * @param roots Vector of roots.
     * @return Polynomial coefficients where coefficient at index \f$ i \f$
     * corresponds to \f$ x^i \f$.
     *
     * Returns polynomial in standard form with coefficients
     * \f$ [c_0, c_1, \ldots, c_{\text{deg}}] \f$ where
     * \f$ p(x) = c_0 + c_1 x + \cdots + c_{\text{deg}} x^{\text{deg}} \f$.
     */
    Eigen::VectorX<Scalar>
    buildPolynomialFromRoots(const Eigen::VectorX<Scalar> &roots) const {
        const Eigen::Index deg = roots.size();
        Eigen::VectorX<Scalar> p = Eigen::VectorX<Scalar>::Zero(deg + 1);
        p(deg) = static_cast<Scalar>(1);

        for (Eigen::Index i = 1; i <= deg; ++i) {
            p.segment(deg - i, i) -= roots(i - 1) * p.tail(i).eval();
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
     * @brief Build quotient polynomials \f$ g_i(x) = \prod_{j \neq i}
     * (x - \text{root}_j) \f$ by direct multiplication.
     * @param roots Vector of roots.
     * @return Matrix where column \f$ i \f$ contains coefficients of \f$ g_i(x)
     * \f$.
     *
     * Each column contains polynomial coefficients in standard form.
     * Builds each g_i by multiplying all roots except root_i, which is
     * more numerically stable than synthetic division.
     */
    Eigen::MatrixX<Scalar>
    buildQuotentPolynomials(const Eigen::VectorX<Scalar> &roots) const {

        const Eigen::Index num_roots = roots.size();
        const Eigen::Index g_deg = num_roots - 1;
        Eigen::MatrixX<Scalar> g(num_roots, num_roots);

        for (Eigen::Index i = 0; i < num_roots; ++i) {
            // Build roots vector excluding root_i
            Eigen::VectorX<Scalar> other_roots(g_deg);
            other_roots.head(i) = roots.head(i);
            other_roots.tail(g_deg - i) = roots.tail(g_deg - i);

            g.col(i) = buildPolynomialFromRoots(other_roots);
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

        // We sort binomials from one corresponding to largest deg to smallest
        Eigen::VectorX<Scalar> binoms(num_coeffs);
        binoms(0) = 1;
        for (Eigen::Index i = 1; i < num_coeffs; ++i) {
            const Scalar idx = static_cast<Scalar>(max_binom_deg - i);
            binoms(i) = -binoms(i - 1) * (idx + 1) / (x_minus_1_deg - idx);
        }

        // Apply to poly (column-wise)
        Eigen::MatrixX<Scalar> poly_new =
            Eigen::MatrixX<Scalar>::Zero(max_deg - min_deg + 1, poly.cols());
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

    /*!
     * @brief Update eigenvalues after rank-1 update.
     * @param lambda_deflated Deflated eigenvalues (diagonal of base matrix).
     * @param q_col Column vector for rank-1 update (may have one extra element
     * compared to lambda_deflated).
     * @return Updated eigenvalues after adding rank-1 update.
     *
     * Computes eigenvalues of diag(lambda_extended) + q_col * q_col^T
     * using secular equation solver, where lambda_extended is lambda_deflated
     * padded with a zero if q_col is longer.
     */
    Eigen::VectorX<Scalar>
    updateEigenvalues(const Eigen::VectorX<Scalar> &lambda_deflated,
                      const Eigen::VectorX<Scalar> &q_col) const {
        const Eigen::Index q_size = q_col.size();
        Eigen::VectorX<Scalar> lambda_extended;

        if (lambda_deflated.size() < q_size) {
            // q_col has one extra element; pad lambda with zero at the
            // beginning
            lambda_extended = Eigen::VectorX<Scalar>::Zero(q_size);
            lambda_extended.tail(lambda_deflated.size()) = lambda_deflated;
        } else {
            lambda_extended = lambda_deflated;
        }

        Utils::SecularEquationSolver<Scalar> solver;
        return solver.solve(lambda_extended, q_col);
    }
};

} // namespace MatSubset

#endif // MAT_SUBSET_DERANDOMIZED_VOLUME_SELECTOR_H