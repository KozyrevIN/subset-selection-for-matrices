#ifndef MAT_SUBSET_DERANDOMIZED_VOLUME_SELECTOR_H
#define MAT_SUBSET_DERANDOMIZED_VOLUME_SELECTOR_H

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
 */
template <typename Scalar>
class DerandomizedVolumeSelector : public SelectorBase<Scalar> {
  public:
    /*!
     * @brief Constructor for `DerandomizedVolumeSelector`.
     * @param tolerance Small tolerance for numerical stability. Default: 1e-8.
     */
    explicit DerandomizedVolumeSelector(
        Scalar tolerance = static_cast<Scalar>(1e-8))
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
        Eigen::VectorX<Scalar> lambda = Eigen::VectorX<Scalar>::Zero(m);
        Eigen::Index r = 0;

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<Scalar>> eigensolver;

        // Main loop
        for (Eigen::Index t = 0; t < k; ++t) {
            // Auxiliary polynomial construction
            Eigen::Index d = n - t + r - m;
            Eigen::Index l = r - std::min(r, d);

            Eigen::VectorX<Scalar> inv_roots =
                lambda.segment(l, r - l).cwiseInverse();
            Eigen::Index roots_deflated = deflateAndCheckOverflow(inv_roots);
            Eigen::VectorX<Scalar> active_roots =
                inv_roots.tail(inv_roots.size() - roots_deflated);

            Eigen::VectorX<Scalar> p = buildP(active_roots);
            // Here i-th row contains coefficients of \f$ g_i \f$
            Eigen::MatrixX<Scalar> g = buildG(active_roots);

            Eigen::Index x_minus_1_deg = d - active_roots.size() - 1;
            const bool r_less_m = (r < m);
            Eigen::Index min_deg_p =
                std::max(n - k - 1 - r_less_m, static_cast<Eigen::Index>(0));
            Eigen::Index max_deg_p;
            if (x_minus_1_deg >= 0) {
                // We need only coefficients \f$ n - k, n - k - 1\f$ and \f$ n -
                // k - 2 \f$ if \f$ r < m \f$.
                max_deg_p = n - k;
                applyBinomials(p, g, x_minus_1_deg, max_deg_p, min_deg_p);
                x_minus_1_deg = 0;
            } else {
                // We need all coefficients from \f$ n - k - 1\f$ (\f$ n -
                // k - 2 \f$ if \f$ r < m \f$) to \f$ d \f$.
                max_deg_p = d;
                const Eigen::Index num_coeffs = max_deg_p - min_deg_p + 1;
                p = p.tail(num_coeffs).eval();
                if (num_coeffs <= g.cols()) {
                    g = g.rightCols(num_coeffs).eval();
                } else {
                    Eigen::MatrixX<Scalar> g_new = Eigen::MatrixX<Scalar>::Zero(g.rows(), num_coeffs);
                    g_new.rightCols(g.cols()) = g;
                    g = g_new;
                }
            }

            // Characteristic polynomial construction
            Eigen::MatrixX<Scalar> Lambda_inv_Q_squared =
                active_roots.asDiagonal() *
                Q.middleRows(l + roots_deflated, r - l).cwiseAbs2();
            Eigen::ArrayX<Scalar> scale_coeffs =
                1 + Lambda_inv_Q_squared.colwise().sum().array();
            Eigen::ArrayXX<Scalar> g_factor = g.transpose() *
                                              active_roots.asDiagonal() *
                                              Lambda_inv_Q_squared;
            Eigen::MatrixX<Scalar> p_S =
                p.replicate(1, n - t) +
                (g_factor.rowwise() / scale_coeffs.transpose()).matrix();

            // Handle additional root and perform division by (x - 1) if needed
            Eigen::VectorX<Scalar> w =
                Q.bottomRows(m - r).colwise().squaredNorm();
            for (Eigen::Index j = 0; j < n - t; ++j) {
                bool needs_division = (x_minus_1_deg < 0);
                bool needs_multiplication = (w(j) > tolerance);
                bool w_is_one = std::abs(w(j) - Scalar(1)) < tolerance;

                // Optimization: if w_j ≈ 1 and we divide by (x-1), they cancel
                if (needs_division && needs_multiplication && w_is_one) {
                    // No operation needed
                    continue;
                }

                if (needs_division) {
                    // Divide by (x - 1)
                    p_S.col(j).tail(p_S.rows() - 1) =
                        dividePolynomialByLinear(p_S.col(j), Scalar(1));
                }

                if (needs_multiplication) {
                    // Multiply by (x - w_j^{-1})
                    Scalar root = static_cast<Scalar>(1) / w(j);
                    p_S.col(j) = multiplyPolynomialByLinear(p_S.col(j), root)
                                     .tail(p_S.rows());
                }
            }

            // Column selection and update
            Eigen::ArrayX<Scalar> c_n_k = p_S.row(max_deg_p - n + k);
            Eigen::ArrayX<Scalar> c_n_k_1 = p_S.row(max_deg_p - n + k + 1);
            Eigen::ArrayX<Scalar> ratio = (c_n_k_1 / c_n_k).abs();
            Eigen::Index s;
            ratio.minCoeff(&s);

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
            lambda = eigensolver.eigenvalues().reverse();
            Q = eigensolver.eigenvectors().rowwise().reverse().transpose() * Q;

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
     * @brief Divide polynomial by (x - root) using synthetic division.
     * @param poly Polynomial coefficients [c_0, c_1, ..., c_n] where
     * poly(x) = c_0 + c_1*x + ... + c_n*x^n.
     * @param root The root value to divide by.
     * @return Quotient polynomial (degree reduced by 1).
     *
     * Assumes poly(root) = 0 or is very small (valid division).
     */
    Eigen::VectorX<Scalar>
    dividePolynomialByLinear(const Eigen::VectorX<Scalar> &poly,
                             Scalar root) const {
        const Eigen::Index deg = poly.size() - 1;
        Eigen::VectorX<Scalar> quotient(deg);

        // Synthetic division: work from highest degree down
        quotient(deg - 1) = poly(deg);
        for (Eigen::Index i = deg - 2; i >= 0; --i) {
            quotient(i) = poly(i + 1) + root * quotient(i + 1);
        }

        return quotient;
    }

    /*!
     * @brief Multiply polynomial by (x - root).
     * @param poly Polynomial coefficients [c_0, c_1, ..., c_n].
     * @param root The root value to multiply by.
     * @return Product polynomial (degree increased by 1).
     */
    Eigen::VectorX<Scalar>
    multiplyPolynomialByLinear(const Eigen::VectorX<Scalar> &poly,
                               Scalar root) const {
        const Eigen::Index deg = poly.size() - 1;
        Eigen::VectorX<Scalar> product(deg + 2);

        product.head(deg + 1) = -root * poly;
        product.tail(deg + 1) += poly;

        return product;
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
     * @brief Count leading roots equal to 1 and check for overflow.
     * @param roots Vector of roots sorted in increasing order, all >= 1.
     * @return Number of leading roots equal to 1 (within tolerance).
     */
    Eigen::Index
    deflateAndCheckOverflow(const Eigen::VectorX<Scalar> &roots) const {
        Eigen::Index roots_deflated = 0;
        while (roots_deflated < roots.size() &&
               std::abs(roots[roots_deflated] - Scalar(1)) < tolerance) {
            roots_deflated++;
        }

        const Eigen::Index deg = roots.size() - roots_deflated;

        const Scalar max_log_val =
            std::log10(std::numeric_limits<Scalar>::max());
        const Scalar safety_buffer = static_cast<Scalar>(deg) / 3;
        const Scalar safe_limit = max_log_val - safety_buffer;

        Scalar log_magnitude = 0;
        for (Eigen::Index i = roots_deflated; i < roots.size(); ++i) {
            log_magnitude += std::log10(roots[i]);
        }

        if (log_magnitude > safe_limit) {
            std::cerr << "Warning: Predicted overflow! Log magnitude sum: "
                      << log_magnitude << std::endl;
        }

        return roots_deflated;
    }

    /*!
     * @brief Build polynomial p(x) from non-deflated roots.
     * @param roots Active roots (after deflation).
     * @return Polynomial coefficients in standard form.
     */
    Eigen::VectorX<Scalar> buildP(const Eigen::VectorX<Scalar> &roots) const {
        return buildPolynomialFromRoots(roots);
    }

    /*!
     * @brief Build quotient polynomials \f$ g_i(x) = \prod_{j \neq i}
     * (x - \text{root}_j) \f$ by direct multiplication.
     * @param roots Vector of roots.
     * @return Matrix where row \f$ i \f$ contains coefficients of \f$ g_i(x)
     * \f$.
     *
     * Each row contains polynomial coefficients in standard form.
     * Builds each g_i by multiplying all roots except root_i, which is
     * more numerically stable than synthetic division.
     */
    Eigen::MatrixX<Scalar> buildG(const Eigen::VectorX<Scalar> &roots) const {
        const Eigen::Index num_roots = roots.size();
        const Eigen::Index g_deg = num_roots - 1;

        Eigen::MatrixX<Scalar> g(num_roots, g_deg + 1);

        for (Eigen::Index i = 0; i < num_roots; ++i) {
            // Build roots vector excluding root_i
            Eigen::VectorX<Scalar> other_roots(g_deg);
            other_roots.head(i) = roots.head(i);
            other_roots.tail(g_deg - i) = roots.tail(g_deg - i);

            g.row(i) = buildPolynomialFromRoots(other_roots).transpose();
        }

        return g;
    }

    /*!
     * @brief Multiply p and g by \f$ (x-1)^{\text{x\_minus\_1\_deg}} \f$ up to
     * a constant, keeping only coefficients for degrees
     * [min_deg, max_deg].
     * @param p Polynomial coefficients (replaced in-place with trimmed result).
     * @param g Quotient polynomial matrix (replaced in-place with trimmed
     * result).
     * @param x_minus_1_deg Exponent of (x-1).
     * @param max_deg Highest degree to keep.
     * @param min_deg Lowest degree to keep.
     */
    void applyBinomials(Eigen::VectorX<Scalar> &p, Eigen::MatrixX<Scalar> &g,
                        Eigen::Index x_minus_1_deg, Eigen::Index max_deg,
                        Eigen::Index min_deg) const {

        const Eigen::Index p_deg = p.size() - 1;

        // We need binomial indices [min_deg - p_deg, max_deg].
        const Eigen::Index min_binom_deg =
            std::max(min_deg - p_deg, static_cast<Eigen::Index>(0));
        const Eigen::Index max_binom_deg = max_deg;
        const Eigen::Index num_coeffs = max_binom_deg - min_binom_deg + 1;

        // We sort binomials from one corresponding to largest deg to smallest
        Eigen::VectorX<Scalar> binoms(num_coeffs);
        binoms(0) = 1;
        for (Eigen::Index i = 1; i < num_coeffs; ++i) {
            const Scalar idx = static_cast<Scalar>(max_binom_deg - i);
            binoms(i) = -binoms(i - 1) * (idx + 1) / (x_minus_1_deg - idx);
        }

        // Apply to p
        Eigen::VectorX<Scalar> p_new(max_deg - min_deg + 1);
        for (Eigen::Index i = 0; i < max_deg - min_deg + 1; ++i) {
            Eigen::Index conv_length = std::min(binoms.size() - i, p.size());
            p_new(p_new.size() - i - 1) =
                p.head(conv_length).dot(binoms.segment(i, conv_length));
        }
        p = p_new;

        // Apply to g
        Eigen::MatrixX<Scalar> g_new(g.rows(), max_deg - min_deg + 1);
        for (Eigen::Index i = 0; i < max_deg - min_deg + 1; ++i) {
            Eigen::Index conv_length = std::min(binoms.size() - i, g.cols());
            g_new.col(i) =
                g.leftCols(conv_length) * binoms.segment(i, conv_length);
        }
        g = g_new;
    }
};

} // namespace MatSubset

#endif // MAT_SUBSET_DERANDOMIZED_VOLUME_SELECTOR_H