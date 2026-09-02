#ifndef MAT_SUBSET_DERANDOMIZED_VOLUME_SELECTOR_H
#define MAT_SUBSET_DERANDOMIZED_VOLUME_SELECTOR_H

#include <algorithm> // For std::max
#include <cassert>   // For std::assert
#include <cmath>     // For std::sqrt
#include <limits>    // For std::numeric_limits

#include <Eigen/Eigenvalues> // For Eigen::SelfAdjointEigenSolver
#include <Eigen/QR>          // For Eigen::HouseholderQR

#include "SelectorBase.h" // Base class

namespace MatSubset {

/*!
 * @brief Approximates subset selection problem for matrices by derandomizing
 * forward volume sampling.
 * @tparam Scalar The underlying scalar type (e.g., `float`, `double`).
 *
 * Implements the forward derandomized volume sampling (FDVS) algorithm. The
 * algorithm greedily selects columns by maintaining the eigendecomposition of
 * \f$ Q_\mathcal{S} Q_\mathcal{S}^T \f$ and scoring each candidate by a ratio
 * of two adjacent coefficients of a characteristic polynomial.
 *
 * @note Every polynomial here is written in the *forward* variable, i.e. with
 * the eigenvalues \f$ \omega_i \in (0, 1) \f$ themselves as roots rather than
 * their reciprocals. All three terms of the rank-one update then carry the
 * same sign (see `selectSubsetImpl`), so the score is computed without
 * cancellation and the coefficients stay bounded by \f$ \binom{l}{s} \f$
 * instead of growing like \f$ \prod_i \omega_i^{-1} \f$.
 *
 * @note The coefficients carry the binomial weights of \f$ (x - 1)^{D} \f$
 * from the moment they are built, rather than meeting them at the end. What
 * the score needs is a handful of coefficients of
 * \f$ (x-1)^{D} \chi(x) \f$, each a sum of products
 * \f$ \chi_a \binom{D}{b - a} \f$. Those products span a modest range, but
 * the two factors do not: with \f$ l \f$ active eigenvalues the low-order
 * \f$ \chi_a \f$ fall off like \f$ \prod_i \omega_i \f$ while the
 * binomials climb like \f$ (D/b)^{a} \f$, each crossing double precision's
 * exponent range near \f$ l \approx 150 \f$ on a matrix with strongly
 * non-uniform leverage. Formed separately, the small factor underflows to zero
 * while the large one is still finite, and the term is lost even though its
 * product is not negligible — the selection then degrades below random. Scaled
 * by \f$ w(a) = \binom{D}{\text{max\_deg} - a} \f$ throughout, every
 * quantity stays in range: the polynomial is built through
 * \f$ \sigma(a) = w(a)/w(a+1) \f$, the quotients through the same ratios,
 * and the three coefficients the score needs come out as one plain sum and two
 * sums weighted by single binomial ratios. Any underflow that survives the
 * scaling is then benign: every term of those sums shares a sign, so what is
 * dropped is negligible relative to what is kept.
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
        const Eigen::Index m = X.rows();
        const Eigen::Index n = X.cols();

        // Thin LQ decomposition of X; Q has orthonormal rows.
        Eigen::HouseholderQR<Eigen::MatrixX<Scalar>> qr(X.transpose());
        Eigen::MatrixX<Scalar> Q =
            (qr.householderQ() * Eigen::MatrixX<Scalar>::Identity(n, m))
                .transpose();

        std::vector<Eigen::Index> selected_indices;
        selected_indices.reserve(static_cast<size_t>(k));
        std::vector<Eigen::Index> remaining_indices(n);
        for (Eigen::Index j = 0; j < n; ++j) {
            remaining_indices[static_cast<size_t>(j)] = j;
        }

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<Scalar>> eigensolver;
        // Eigenvalues of Q_S Q_S^T, kept in non-increasing order: the block of
        // ones comes first, then the active block, then the zeros.
        Eigen::VectorX<Scalar> lambda = Eigen::VectorX<Scalar>::Zero(m);

        // Main loop
        for (Eigen::Index t = 0; t < k; ++t) {
            // Notation
            Eigen::Index ones = 0;
            while (ones < m && lambda(ones) > Scalar(1) - tolerance) {
                ++ones;
            }
            Eigen::Index r = ones;
            while (r < m && lambda(r) > tolerance) {
                ++r;
            }
            const Eigen::Index l = r - ones; // active eigenvalues, in (0, 1)
            const Eigen::Index d = n - t + r - m;

            // Q_R restricted to the non-ones rows has m - ones rows and n - t
            // columns and full row rank, so m - ones <= n - t, i.e. l <= d.
            assert(l <= d && "(x - 1) exponent cannot drop below -1");
            const bool delta = (l < d);
            const Eigen::Index binom_deg = d - l - (delta ? 1 : 0);

            // B holds the rows of Q spanned by the active eigenvalues;
            // c_sq(j) = ||c_j||^2 is the mass of column j outside range(Q_S).
            Eigen::MatrixX<Scalar> B_sq = Q.middleRows(ones, l).cwiseAbs2();
            Eigen::RowVectorX<Scalar> c_sq =
                Eigen::RowVectorX<Scalar>::Zero(n - t);
            if (r < m) {
                c_sq = Q.bottomRows(m - r).colwise().squaredNorm();
            }
            Eigen::VectorX<Scalar> omega = lambda.segment(ones, l);

            // Auxiliary polynomial construction. We need the coefficients of
            //   p(x)   = (x - 1)^binom_deg prod_i (x - omega_i),
            //   g_i(x) = p(x) / (x - omega_i),
            // for degrees [min_deg, max_deg], all up to one common constant.
            const Eigen::Index min_deg = d - n + k - 1;
            const Eigen::Index max_deg = d - n + k + 1;

            Eigen::MatrixX<Scalar> p, g;
            if (max_deg <= binom_deg) {
                // The weighted path: coefficient a is carried scaled by
                // w(a) = C(binom_deg, max_deg - a), which is exactly the
                // binomial it meets in the sums below.
                const Eigen::VectorX<Scalar> sigma =
                    binomialWeightRatios(binom_deg, max_deg, l);
                const Eigen::VectorX<Scalar> chi =
                    buildPolynomialFromRoots(omega, sigma);
                const Eigen::MatrixX<Scalar> chi_quotients =
                    buildQuotientPolynomials(chi, omega, sigma);
                // Row 2 is a plain sum — its binomial is already in the
                // coefficients — and rows 0 and 1 are the same sum reweighted
                // by one and two binomial ratios.
                const Eigen::MatrixX<Scalar> weights =
                    collapseWeights(binom_deg, min_deg, max_deg, l);
                p = weights * chi;
                g = weights.leftCols(l) * chi_quotients;
            } else {
                // max_deg > binom_deg only when all but a handful of columns
                // are being selected (it needs k > n - m - 2), which bounds
                // (x - 1)^binom_deg to degree max_deg <= k - t + 1 and its
                // coefficients to a range no scaling is needed for. Weighting
                // cannot be anchored at max_deg there — the binomial does not
                // reach that far — so the coefficients are built unweighted
                // (sigma = 1) and meet the binomials at the end, as before.
                const Eigen::VectorX<Scalar> sigma =
                    Eigen::VectorX<Scalar>::Ones(l);
                const Eigen::VectorX<Scalar> chi =
                    buildPolynomialFromRoots(omega, sigma);
                const Eigen::MatrixX<Scalar> chi_quotients =
                    buildQuotientPolynomials(chi, omega, sigma);
                p = applyBinomials(chi, binom_deg, min_deg, max_deg);
                g = applyBinomials(chi_quotients, binom_deg, min_deg, max_deg);
            }

            // Characteristic polynomial construction. Row 0 and row 1 of p_j
            // hold the coefficients of degree d - n + k and d - n + k + 1.
            Eigen::MatrixX<Scalar> p_j(2, n - t);
            if (delta) {
                // p_j(x) = (x - ||c_j||^2) p(x) - x sum_i B_ij^2 g_i(x).
                //
                // p and g have alternating coefficients of opposite parity, so
                // all three terms below share a sign: nothing cancels.
                for (Eigen::Index row = 0; row < 2; ++row) {
                    p_j.row(row) =
                        Eigen::RowVectorX<Scalar>::Constant(n - t, p(row, 0)) -
                        p(row + 1, 0) * c_sq - g.row(row) * B_sq;
                }
            } else {
                // binom_deg would be -1 here, so the bracket above carries an
                // extra root at x = 1. Dividing it out in closed form, using
                // 1 - ||c_j||^2 = sum_i B_ij^2 / (1 - omega_i), leaves
                // p_j(x) = p(x) + sum_i B_ij^2 omega_i / (1 - omega_i) g_i(x).
                Eigen::MatrixX<Scalar> weighted_B_sq =
                    (omega.array() / (1 - omega.array()))
                        .matrix()
                        .asDiagonal() *
                    B_sq;
                for (Eigen::Index row = 0; row < 2; ++row) {
                    p_j.row(row) = Eigen::RowVectorX<Scalar>::Constant(
                                       n - t, p(row + 1, 0)) +
                                   g.row(row + 1) * weighted_B_sq;
                }
            }

            // Greedy selection and update
            Eigen::ArrayX<Scalar> ratios =
                (p_j.row(1).array() / p_j.row(0).array()).abs();
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
            // Reverse to keep the eigenvalues in non-increasing order.
            lambda = eigensolver.eigenvalues().reverse();
            Q = eigensolver.eigenvectors().rowwise().reverse().transpose() * Q;
        }

        return selected_indices;
    }

  private:
    /*!
     * @brief Numerical tolerance for deflating eigenvalues equal to 0 or 1.
     */
    Scalar tolerance;

    /*!
     * @brief Ratios \f$ \sigma(a) = w(a)/w(a+1) \f$ of the binomial weights
     * \f$ w(a) = \binom{D}{\text{max\_deg} - a} \f$ the coefficients are
     * carried in.
     * @param binom_deg The exponent \f$ D \f$ of \f$ (x - 1)^{D} \f$.
     * @param max_deg The highest output degree, where the weight is anchored:
     * \f$ w(0) = \binom{D}{\text{max\_deg}} \f$. Must not exceed
     * \f$ D \f$.
     * @param deg Degree of the polynomial being weighted.
     * @return \f$ \sigma(0 \ldots \deg - 1) \f$.
     *
     * Only the ratios are ever formed, never the weights themselves: they are
     * \f$ \binom{D}{b} / \binom{D}{b - 1} = (D - b + 1)/b \f$ at
     * \f$ b = \text{max\_deg} - a \f$, each a modest number, where the
     * weights they compose are not representable. Past the end of the binomial
     * (\f$ b \le 0 \f$) the denominator is clamped to 1: those coefficients
     * meet a zero binomial in `collapseWeights` and contribute nothing, and
     * the clamp only keeps the continuation finite for the recurrences that
     * pass through them.
     */
    Eigen::VectorX<Scalar> binomialWeightRatios(Eigen::Index binom_deg,
                                                Eigen::Index max_deg,
                                                Eigen::Index deg) const {
        Eigen::VectorX<Scalar> sigma(deg);
        for (Eigen::Index a = 0; a < deg; ++a) {
            const Eigen::Index b = max_deg - a;
            sigma(a) = static_cast<Scalar>(binom_deg - b + 1) /
                       static_cast<Scalar>(std::max<Eigen::Index>(b, 1));
        }
        return sigma;
    }

    /*!
     * @brief Rescale a polynomial so its largest coefficient sits well inside
     * the exponent range.
     * @param poly Coefficients, rescaled in place.
     *
     * The factor is global, and everything downstream of it — the quotients,
     * the collapsed coefficients, the score's ratio — is homogeneous in it, so
     * this changes no result. It only stops the weighted coefficients, which
     * climb steadily as the polynomial fills in, from running out of range on
     * the way. The bounds are the square roots of the type's own limits, so a
     * rescale is rare and leaves half the exponent range as headroom either
     * side.
     */
    void rescale(Eigen::VectorX<Scalar> &poly) const {
        const Scalar high = std::sqrt(std::numeric_limits<Scalar>::max());
        const Scalar low = std::sqrt(std::numeric_limits<Scalar>::min());

        const Scalar peak = poly.cwiseAbs().maxCoeff();
        if (peak > high || (peak > static_cast<Scalar>(0) && peak < low)) {
            poly /= peak;
        }
    }

    /*!
     * @brief Build the weighted polynomial
     * \f$ P_a = w(a) [x^a] \prod_i (x - \text{root}_i) \f$ via incremental
     * left-to-right multiplication, consuming roots in ascending-magnitude
     * order.
     * @param roots Vector of roots in \f$ [0, 1) \f$, **assumed sorted in
     * descending order** (which is what the caller produces by slicing the
     * non-increasing `lambda`).
     * @param sigma Weight ratios \f$ w(a)/w(a+1) \f$ from
     * `binomialWeightRatios`; all ones builds the plain polynomial.
     * @return Weighted coefficients in standard form (size = deg + 1), up to
     * one global constant.
     *
     * Because every root is non-negative the coefficients strictly alternate
     * in sign, and each step \f$ p \gets (x - r_i) p \f$ combines two
     * like-signed terms. The construction is therefore free of cancellation.
     * In weighted form the step \f$ p_a \gets p_a - r p_{a+1} \f$ becomes
     * \f$ P_a \gets P_a - r \sigma(a) P_{a+1} \f$: one extra elementwise
     * multiply, and the weights themselves are never formed.
     */
    Eigen::VectorX<Scalar>
    buildPolynomialFromRoots(const Eigen::VectorX<Scalar> &roots,
                             const Eigen::VectorX<Scalar> &sigma) const {
        const Eigen::Index deg = roots.size();
        Eigen::VectorX<Scalar> p = Eigen::VectorX<Scalar>::Zero(deg + 1);
        p(deg) = static_cast<Scalar>(1);

        // Consume roots smallest-magnitude first: roots(deg - i) walks the
        // input in reverse (the caller passes a descending vector).
        for (Eigen::Index i = 1; i <= deg; ++i) {
            p.segment(deg - i, i).array() -=
                roots(deg - i) *
                (sigma.segment(deg - i, i).array() * p.tail(i).array()).eval();
            rescale(p);
        }

        return p;
    }

    /*!
     * @brief Build the weighted quotient polynomials
     * \f$ g_i(x) = p(x) / (x - r_i) \f$ via composite (two-way) synthetic
     * division, in the same weights as \f$ p \f$.
     * @param p Weighted coefficients of \f$ p(x) = \prod_i (x - r_i) \f$,
     * standard form (size = num_roots + 1).
     * @param roots Vector of roots, **assumed sorted in descending order**.
     * @param sigma The weight ratios \f$ p \f$ is carried in.
     * @return Matrix where column \f$ i \f$ contains the weighted
     * coefficients of \f$ g_i(x) \f$ in standard form (size = num_roots).
     *
     * Cost: \f$ O(\text{num\_roots}^2) \f$ instead of the naive
     * \f$ O(\text{num\_roots}^3) \f$ that would result from building each
     * column independently.
     *
     * Stability strategy: coefficient \f$ b_a \f$ of \f$ g_i \f$ equals
     * \f$ \pm e_{\deg - a}(\text{roots} \setminus \{r_i\}) \f$, and the
     * forward step \f$ b_{a-1} = c_a + r_i b_a \f$ evaluates
     * \f$ e_s = e_s(\text{all}) - r_i e_{s-1} \f$. That subtraction is benign
     * exactly while the roots being absorbed are larger than \f$ r_i \f$, and
     * the backward step is benign exactly while they are smaller. Since
     * `roots` is descending, precisely \f$ i \f$ roots exceed \f$ r_i \f$ and
     * the crossover sits at degree \f$ \deg - i \f$: each recurrence is run
     * only over the range where it is stable.
     *
     * Both recurrences carry the weights the same way the build does — a
     * factor \f$ \sigma \f$ where they step down in degree, its reciprocal
     * where they step up — so the quotients come out weighted like \f$ p \f$,
     * as the sums that consume them require.
     */
    Eigen::MatrixX<Scalar>
    buildQuotientPolynomials(const Eigen::VectorX<Scalar> &p,
                             const Eigen::VectorX<Scalar> &roots,
                             const Eigen::VectorX<Scalar> &sigma) const {
        const Eigen::Index num_roots = roots.size();
        const Eigen::Index g_deg = num_roots - 1;
        Eigen::MatrixX<Scalar> g(num_roots, num_roots);

        for (Eigen::Index i = 0; i < num_roots; ++i) {
            const Scalar r = roots(i);
            const Eigen::Index meet = g_deg - i;

            // Forward recurrence (high-to-low):
            // B_{a-1} = sigma(a-1) * (C_a + r * B_a).
            Scalar fwd = sigma(g_deg) * p(num_roots); // b_{g_deg}
            g(g_deg, i) = fwd;
            for (Eigen::Index a = g_deg; a > meet; --a) {
                fwd = sigma(a - 1) * (p(a) + r * fwd);
                g(a - 1, i) = fwd;
            }

            // Backward recurrence (low-to-high):
            // B_{a+1} = (B_a / sigma(a) - C_{a+1}) / r.
            if (meet > 0) {
                Scalar bwd = -p(0) / r; // b_0
                g(0, i) = bwd;
                for (Eigen::Index a = 0; a + 1 < meet; ++a) {
                    bwd = (bwd / sigma(a) - p(a + 1)) / r;
                    g(a + 1, i) = bwd;
                }
            }
        }

        return g;
    }

    /*!
     * @brief The weights that collapse a weighted polynomial into the three
     * coefficients of \f$ (x - 1)^{D} p(x) \f$ the score needs.
     * @param binom_deg The exponent \f$ D \f$ of \f$ (x - 1)^{D} \f$.
     * @param min_deg Lowest output degree (may be negative, giving a zero row).
     * @param max_deg Highest output degree; the weights are anchored here and
     * it must not exceed \f$ D \f$.
     * @param deg Degree of the polynomial being collapsed.
     * @return A \f$ 3 \times (\deg + 1) \f$ matrix; multiplying it by the
     * weighted coefficients gives the output degrees `min_deg`, `min_deg + 1`
     * and `max_deg`, in that order, up to one global constant.
     *
     * Coefficient \f$ a \f$ is carried scaled by
     * \f$ w(a) = \binom{D}{\text{max\_deg} - a} \f$, which is precisely
     * the binomial the top row wants, so that row is a plain sum. The other
     * two want the next binomials down, and
     * \f$ \binom{D}{b-1} / \binom{D}{b} = b / (D - b + 1) \f$ supplies
     * each as one modest factor. The alternating signs of \f$ (x - 1)^{D} \f$
     * and of the coefficients cancel to leave every term of a row sharing one
     * sign, which is what makes the sums cancellation-free; the sign that
     * alternates between rows is kept, since the score's numerator and
     * denominator rely on it.
     */
    Eigen::MatrixX<Scalar> collapseWeights(Eigen::Index binom_deg,
                                           Eigen::Index min_deg,
                                           Eigen::Index max_deg,
                                           Eigen::Index deg) const {
        Eigen::MatrixX<Scalar> weights =
            Eigen::MatrixX<Scalar>::Zero(max_deg - min_deg + 1, deg + 1);

        for (Eigen::Index a = 0; a <= deg; ++a) {
            const Eigen::Index b = max_deg - a;
            // Past the end of the binomial nothing contributes, and b only
            // falls from here.
            if (b < 0) {
                break;
            }
            const Scalar sign = (a % 2 == 0) ? static_cast<Scalar>(1)
                                             : static_cast<Scalar>(-1);
            // Ratios to the binomials one and two degrees below the anchor.
            const Scalar ratio_mid =
                (b > 0) ? static_cast<Scalar>(b) /
                              static_cast<Scalar>(binom_deg - b + 1)
                        : static_cast<Scalar>(0);
            const Scalar ratio_low =
                (b > 1) ? ratio_mid * static_cast<Scalar>(b - 1) /
                              static_cast<Scalar>(binom_deg - b + 2)
                        : static_cast<Scalar>(0);

            weights(0, a) = sign * ratio_low;
            weights(1, a) = -sign * ratio_mid;
            weights(2, a) = sign;
        }

        return weights;
    }

    /*!
     * @brief Multiply polynomial matrix by \f$ (x-1)^{\text{x\_minus\_1\_deg}}
     * \f$ up to a constant, keeping only coefficients for degrees [min_deg,
     * max_deg].
     * @param poly Polynomial matrix where each column is a polynomial.
     * @param x_minus_1_deg Exponent of (x-1).
     * @param min_deg Lowest degree to keep (may be negative; those rows are
     * left at zero).
     * @param max_deg Highest degree to keep.
     * @return New matrix with trimmed polynomial coefficients.
     *
     * @note The normalization constant depends only on `x_minus_1_deg` and
     * `max_deg`, so calls sharing those arguments stay mutually consistent.
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
