#ifndef MAT_SUBSET_GAPPY_POD_SELECTOR_H
#define MAT_SUBSET_GAPPY_POD_SELECTOR_H

#include <Eigen/Eigenvalues> // For Eigen::SelfAdjointEigenSolver

#include "Enums.h"              // For MatSubset::Initialization
#include "VolumePivotingBase.h" // Base class

namespace MatSubset {

/*!
 * @brief Approximates subset selection problem for matrices by the GappyPOD+E
 * algorithm, i.e. column-pivoted QR followed by greedy eigenvalue-based
 * oversampling.
 * @tparam Scalar The underlying scalar type (e.g., `float`, `double`).
 *
 * This selector implements Algorithm 1 of Peherstorfer, Drmac, Gugercin (2020)
 * "Stability of discrete empirical interpolation and gappy proper orthogonal
 * decomposition with randomized and deterministic sampling points". The first
 * \f$ m \f$ points are the Q-DEIM ones: column-pivoted QR of the orthonormal
 * row-space basis \f$ U^{T} = V \f$ of \f$ X \f$, see `QdeimSelector`. The
 * remaining \f$ k - m \f$ points are then added one at a time, each chosen to
 * maximize the growth of the smallest eigenvalue of the Gram matrix
 * \f$ G = U_{\mathcal{S}}^{T} U_{\mathcal{S}} \f$ of the sampled rows — which
 * is the same as maximizing \f$ \sigma_{\min}(X_{\mathcal{S}}) \f$ and hence
 * directly minimizing \f$ \lVert X_{\mathcal{S}}^{\dag} \rVert_{2} \f$.
 *
 * Adding row \f$ u_j \f$ turns \f$ G \f$ into \f$ G + u_j u_j^{T} \f$. Writing
 * \f$ g = \sigma_{m-1}^{2} - \sigma_{m}^{2} \f$ for the gap between the two
 * smallest eigenvalues of \f$ G \f$ and \f$ w \f$ for the eigenvector of the
 * smallest one, the paper scores each candidate by
 * \f[
 *   r_j = \left(g + \lVert u_j \rVert^{2}\right) -
 *         \sqrt{\left(g + \lVert u_j \rVert^{2}\right)^{2} -
 *               4 g \left(w^{T} u_j\right)^{2}},
 * \f]
 * which is twice the exact increase of the smallest eigenvalue in the
 * two-dimensional invariant subspace of the two smallest eigenvalues, and
 * takes the unselected \f$ j \f$ with the largest score. The scaling by two is
 * irrelevant, as the score is only used for ranking.
 *
 * Unlike `DeimSelector` and `QdeimSelector`, which produce one interpolation
 * point per basis vector, this selector supports the oversampled regime
 * \f$ k > m \f$; for \f$ k = m \f$ it degenerates to Q-DEIM. It is the natural
 * point of comparison for the oversampling volume-based selectors of this
 * library, `RectMaxvolSelector` in particular, as it pursues the same quantity
 * \f$ \lVert X_{\mathcal{S}}^{\dag} \rVert \f$ by a spectral rather than a
 * volume criterion.
 *
 * @note The criterion is a local one — the eigenvalue increase is exact only
 * within the two-dimensional subspace it is derived in — so no bound on
 * \f$ \lVert X_{\mathcal{S}}^{\dag} \rVert \f$ is proven for the algorithm and
 * `boundImpl` is left at its default of \f$ 0 \f$.
 */
template <typename Scalar>
class GappyPodSelector : public VolumePivotingBase<Scalar> {
  public:
    /*!
     * @brief Default constructor for `GappyPodSelector`.
     */
    GappyPodSelector() = default;

    /*!
     * @brief Gets the human-readable name of the algorithm.
     * @return The string "gappy pod+e".
     */
    std::string getAlgorithmName() const override { return "gappy pod+e"; }

  protected:
    /*!
     * @brief Core implementation for selecting a subset of \f$ k \f$ columns.
     * @param X The input matrix (dimensions \f$ m \times n \f$) from which
     * columns are to be selected. It is assumed that \f$ X \f$ is full rank
     * for theoretical guarantees.
     * @param k The number of columns to select.
     * @param swap_count Unused; GappyPOD+E only appends points and performs no
     * exchanges, so the caller's -1 ("not tracked") is left in place.
     * @return A `std::vector` of `Eigen::Index` containing the 0-based indices
     * of the selected columns.
     */
    std::vector<Eigen::Index>
    selectSubsetImpl(const Eigen::MatrixX<Scalar> &X, Eigen::Index k,
                     [[maybe_unused]] Eigen::Index *swap_count) override {

        const Eigen::Index m = X.rows();
        const Eigen::Index n = X.cols();

        // Lines 2-3: the first m points are the CPQR pivots. The CPQR strategy
        // stops right after those m pivots, which is exactly Q-DEIM, and
        // overwrites its argument with the orthonormal row-space basis V
        // permuted alongside the returned indices: V.col(i) is the basis
        // vector u_{indices[i]}, and positions i >= m hold the candidates.
        Eigen::MatrixX<Scalar> V = X;
        std::vector<Eigen::Index> indices =
            VolumePivotingBase<Scalar>::selectStartingSet(V, k,
                                                          Initialization::CPQR);

        // Squared norms ||u_j||^2 of the candidate rows. They do not change as
        // points are added, so they are computed once and permuted along with
        // the columns of V.
        Eigen::ArrayX<Scalar> s = V.colwise().squaredNorm();

        // The Gram matrix of the sampled rows. Its eigenvalues are the squared
        // singular values of the selected submatrix and its eigenvectors are
        // the right singular vectors, so the m x m eigendecomposition below
        // replaces the paper's svd(U(p, :)) and each added point is a rank-one
        // update rather than a fresh factorization. Only the lower triangle is
        // ever formed: it is the only part `SelfAdjointEigenSolver` reads and
        // the only part `rankUpdate` writes.
        Eigen::MatrixX<Scalar> G(m, m);
        G.template triangularView<Eigen::Lower>() =
            V.leftCols(m) * V.leftCols(m).transpose();

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<Scalar>> solver(m);

        // Scratch, allocated once; only the leading n - i entries are live.
        Eigen::VectorX<Scalar> b(n);
        Eigen::ArrayX<Scalar> r(n);

        for (Eigen::Index i = m; i < k; ++i) {

            const Eigen::Index candidates = n - i;
            Eigen::Index j_max;

            if (m == 1) {
                // A single basis vector leaves no second eigenvalue to form
                // the gap g with. There the update is exact and trivial:
                // adding row u_j raises the lone eigenvalue by ||u_j||^2.
                s.tail(candidates).maxCoeff(&j_max);
            } else {
                // Line 5: eigenvalues come out in increasing order, so index 0
                // is the smallest one and its eigenvector is the paper's last
                // right singular vector W(:, end).
                solver.compute(G);
                const Scalar g =
                    solver.eigenvalues()(1) - solver.eigenvalues()(0);
                const auto w = solver.eigenvectors().col(0);

                // Lines 7-9. The discriminant is nonnegative in exact
                // arithmetic, as (w^T u_j)^2 <= ||u_j||^2 makes it at least
                // (g - ||u_j||^2)^2, so the clamp only absorbs rounding.
                b.head(candidates).noalias() =
                    V.rightCols(candidates).transpose() * w;
                const auto t = g + s.tail(candidates);
                r.head(candidates) =
                    t - (t.square() - static_cast<Scalar>(4) * g *
                                          b.head(candidates).array().square())
                            .max(static_cast<Scalar>(0))
                            .sqrt();

                // Lines 10-15: the largest score among the candidates. A tie —
                // g = 0, i.e. a repeated smallest singular value, makes every
                // score vanish — resolves to the first candidate, matching the
                // paper's descending sort.
                r.head(candidates).maxCoeff(&j_max);
            }

            j_max += i;

            // Move the chosen point into the selected block, keeping V, s and
            // indices consistently permuted.
            std::swap(indices[static_cast<size_t>(i)],
                      indices[static_cast<size_t>(j_max)]);
            V.col(i).swap(V.col(j_max));
            std::swap(s(i), s(j_max));

            G.template selfadjointView<Eigen::Lower>().rankUpdate(V.col(i));
        }

        indices.resize(static_cast<size_t>(k));
        return indices;
    }
};

} // namespace MatSubset

#endif // MAT_SUBSET_GAPPY_POD_SELECTOR_H
