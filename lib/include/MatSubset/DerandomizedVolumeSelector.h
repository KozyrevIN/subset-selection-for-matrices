#ifndef MAT_SUBSET_DERANDOMIZED_VOLUME_SELECTOR_H
#define MAT_SUBSET_DERANDOMIZED_VOLUME_SELECTOR_H

#include <cassert>
#include <cmath>  // For std::sqrt
#include <limits> // For std::numeric_limits

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
            const Eigen::Index l = r - ones;     // active eigenvalues, in (0, 1)
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

            Eigen::VectorX<Scalar> chi = buildPolynomialFromRoots(omega);
            Eigen::MatrixX<Scalar> chi_quotients =
                buildQuotientPolynomials(chi, omega);
            Eigen::MatrixX<Scalar> p =
                applyBinomials(chi, binom_deg, min_deg, max_deg);
            Eigen::MatrixX<Scalar> g =
                applyBinomials(chi_quotients, binom_deg, min_deg, max_deg);

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
     * @brief Build polynomial \f$ p(x) = \prod_i (x - \text{root}_i) \f$ via
     * incremental left-to-right multiplication, consuming roots in
     * ascending-magnitude order.
     * @param roots Vector of roots in \f$ [0, 1) \f$, **assumed sorted in
     * descending order** (which is what the caller produces by slicing the
     * non-increasing `lambda`).
     * @return Polynomial coefficients in standard form (size = deg + 1).
     *
     * Because every root is non-negative the coefficients strictly alternate
     * in sign, and each step \f$ p \gets (x - r_i) p \f$ combines two
     * like-signed terms. The construction is therefore free of cancellation
     * and the coefficients are bounded by \f$ \binom{\deg}{s} \f$.
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
     * @brief Build quotient polynomials \f$ g_i(x) = p(x) / (x - r_i) \f$ via
     * composite (two-way) synthetic division.
     * @param p Coefficients of \f$ p(x) = \prod_i (x - r_i) \f$, standard form
     * (size = num_roots + 1).
     * @param roots Vector of roots, **assumed sorted in descending order**.
     * @return Matrix where column \f$ i \f$ contains coefficients of
     * \f$ g_i(x) \f$ in standard form (size = num_roots).
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
     */
    Eigen::MatrixX<Scalar>
    buildQuotientPolynomials(const Eigen::VectorX<Scalar> &p,
                             const Eigen::VectorX<Scalar> &roots) const {
        const Eigen::Index num_roots = roots.size();
        const Eigen::Index g_deg = num_roots - 1;
        Eigen::MatrixX<Scalar> g(num_roots, num_roots);

        for (Eigen::Index i = 0; i < num_roots; ++i) {
            const Scalar r = roots(i);
            const Eigen::Index meet = g_deg - i;

            // Forward recurrence (high-to-low): b_{a-1} = c_a + r * b_a.
            Scalar fwd = p(num_roots); // b_{g_deg}
            g(g_deg, i) = fwd;
            for (Eigen::Index a = g_deg; a > meet; --a) {
                fwd = p(a) + r * fwd;
                g(a - 1, i) = fwd;
            }

            // Backward recurrence (low-to-high): b_{a+1} = (b_a - c_{a+1}) / r.
            if (meet > 0) {
                Scalar bwd = -p(0) / r; // b_0
                g(0, i) = bwd;
                for (Eigen::Index a = 0; a + 1 < meet; ++a) {
                    bwd = (bwd - p(a + 1)) / r;
                    g(a + 1, i) = bwd;
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
