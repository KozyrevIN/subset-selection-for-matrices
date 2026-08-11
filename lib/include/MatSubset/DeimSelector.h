#ifndef MAT_SUBSET_DEIM_SELECTOR_H
#define MAT_SUBSET_DEIM_SELECTOR_H

#include <Eigen/LU> // For Eigen::FullPivLU
#include <Eigen/QR> // For Eigen::HouseholderQR

#include "SelectorBase.h" // Base class

namespace MatSubset {

/*!
 * @brief Approximates subset selection problem for matrices by the Discrete
 * Empirical Interpolation Method (DEIM).
 * @tparam Scalar The underlying scalar type (e.g., `float`, `double`).
 *
 * This selector implements Algorithm 1 of Chaturantabut, Sorensen (2010)
 * "Nonlinear model reduction via discrete empirical interpolation". Working on
 * an orthonormal basis \f$ \{u_\ell\}_{\ell=1}^{m} \f$ of the row space of
 * \f$ X \f$, it picks the first index at the largest entry of \f$ u_1 \f$ and
 * then, for each subsequent basis vector, interpolates it on the indices
 * already chosen and picks the index of the largest entry of the residual
 * \f$ r = u_\ell - Uc \f$. The residual is exactly zero on every selected
 * index, so a fresh index is produced at every step.
 *
 * DEIM is the classical interpolation-point heuristic of model order reduction
 * and serves here as a common baseline. Its error bound
 * \f$ \lVert X_{\mathcal{S}}^{\dag} \rVert \f$ grows with the basis size in
 * the worst case, so it is typically weaker than the volume-based selectors of
 * this library, but it is cheap and requires no oversampling.
 *
 * @note Unlike most selectors here, DEIM produces exactly \f$ m \f$ indices:
 * one interpolation point per basis vector. It therefore only supports
 * \f$ k = m \f$. For the oversampled regime \f$ k > m \f$ use Q-DEIM (which is
 * plain column-pivoted QR, available through the `Initialization::CPQR`
 * strategy of `DominantSelector` or `VolumeAddRemoveSelector`) or one of the
 * volume-based selectors.
 */
template <typename Scalar> class DeimSelector : public SelectorBase<Scalar> {
  public:
    /*!
     * @brief Default constructor for `DeimSelector`.
     */
    DeimSelector() = default;

    /*!
     * @brief Gets the human-readable name of the algorithm.
     * @return The string "deim".
     */
    std::string getAlgorithmName() const override { return "deim"; }

  protected:
    /*!
     * @brief Core implementation for selecting a subset of \f$ k \f$ columns.
     * @param X The input matrix (dimensions \f$ m \times n \f$) from which
     * columns are to be selected. It is assumed that \f$ X \f$ is full rank
     * for theoretical guarantees.
     * @param k The number of columns to select. Must satisfy \f$ k = m \f$.
     * @param swap_count Unused; DEIM performs no exchanges and leaves the
     * caller's -1 ("not tracked") in place.
     * @return A `std::vector` of `Eigen::Index` containing the 0-based indices
     * of the selected columns.
     */
    std::vector<Eigen::Index>
    selectSubsetImpl(const Eigen::MatrixX<Scalar> &X, Eigen::Index k,
                     [[maybe_unused]] Eigen::Index *swap_count) override {

        const Eigen::Index m = X.rows();
        const Eigen::Index n = X.cols();

        assert(k == m && "DeimSelector: DEIM returns one interpolation point "
                         "per basis vector, so only k = m is supported.");

        // LQ decomposition, matching the other selectors: the rows of V are an
        // orthonormal basis of the row space of X, so V.row(l) plays the role
        // of the paper's basis vector u_l and its entries are indexed by the
        // columns we are selecting.
        Eigen::HouseholderQR<Eigen::MatrixX<Scalar>> qr(X.transpose());
        Eigen::MatrixX<Scalar> V =
            (qr.householderQ() * Eigen::MatrixX<Scalar>::Identity(n, m))
                .transpose();

        std::vector<Eigen::Index> indices;
        indices.reserve(static_cast<size_t>(m));

        // Line 1: the first point is the largest entry of the first basis
        // vector.
        Eigen::Index p;
        V.row(0).cwiseAbs().maxCoeff(&p);
        indices.push_back(p);

        // Lines 3-8: interpolate u_l on the points chosen so far and take the
        // largest entry of the residual.
        for (Eigen::Index l = 1; l < m; ++l) {
            // U is the basis so far restricted to nothing (all n entries), and
            // P^T U is that basis sampled at the selected points.
            const Eigen::MatrixX<Scalar> U = V.topRows(l).transpose();
            const Eigen::MatrixX<Scalar> PU = U(indices, Eigen::all);
            const Eigen::VectorX<Scalar> Pu =
                V.row(l).transpose()(indices, Eigen::all);

            // Line 4: solve (P^T U) c = P^T u_l. The system is l x l and
            // nonsingular whenever the basis is linearly independent, but a
            // rank-revealing solve keeps a rank-deficient input from producing
            // garbage rather than an exception.
            const Eigen::VectorX<Scalar> c =
                Eigen::FullPivLU<Eigen::MatrixX<Scalar>>(PU).solve(Pu);

            // Lines 5-6: the residual vanishes on the selected indices by
            // construction, so its argmax is always a new index.
            const Eigen::VectorX<Scalar> r = V.row(l).transpose() - U * c;
            r.cwiseAbs().maxCoeff(&p);
            indices.push_back(p);
        }

        return indices;
    }
};

} // namespace MatSubset

#endif // MAT_SUBSET_DEIM_SELECTOR_H
