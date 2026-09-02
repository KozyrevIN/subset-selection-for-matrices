#ifndef MAT_SUBSET_QDEIM_SELECTOR_H
#define MAT_SUBSET_QDEIM_SELECTOR_H

#include <cassert> // For assert

#include "Enums.h"              // For MatSubset::Initialization
#include "VolumePivotingBase.h" // Base class

namespace MatSubset {

/*!
 * @brief Approximates subset selection problem for matrices by the Q-DEIM
 * algorithm, i.e. column-pivoted QR of the orthonormal row-space basis.
 * @tparam Scalar The underlying scalar type (e.g., `float`, `double`).
 *
 * This selector implements Q-DEIM of Drmac, Gugercin (2016) "A new selection
 * operator for the discrete empirical interpolation method". Where
 * `DeimSelector` builds its index set by interpolating one basis vector at a
 * time, Q-DEIM simply runs column-pivoted QR on the orthonormal basis and
 * keeps the pivots. That is both cheaper and better behaved: it improves the
 * DEIM error constant \f$ \lVert X_{\mathcal{S}}^{\dag} \rVert \f$ from a
 * bound that grows exponentially in the basis size to one that grows like
 * \f$ \mathcal{O}(\sqrt{n - m}\,2^{m}) \f$ in the worst case and is small in
 * practice.
 *
 * The pivoting itself is the first phase of `VolumePivotingBase`, which the
 * volume-based selectors of this library use to obtain their starting set.
 * Q-DEIM is exactly that phase used on its own, with no volume refinement
 * afterwards — which is what distinguishes it from `DominantSelector` and
 * `VolumeAddRemoveSelector` constructed with `Initialization::CPQR`, as those
 * go on to run their exchange loops.
 *
 * @note Like `DeimSelector`, Q-DEIM produces one interpolation point per basis
 * vector, so only \f$ k = m \f$ is supported. For the oversampled regime
 * \f$ k > m \f$ see `GappyPodSelector`, which keeps these pivots as its
 * starting set and greedily appends the remaining \f$ k - m \f$ points.
 */
template <typename Scalar>
class QdeimSelector : public VolumePivotingBase<Scalar> {
  public:
    /*!
     * @brief Default constructor for `QdeimSelector`.
     */
    QdeimSelector() = default;

    /*!
     * @brief Gets the human-readable name of the algorithm.
     * @return The string "qdeim".
     */
    std::string getAlgorithmName() const override { return "qdeim"; }

  protected:
    /*!
     * @brief Core implementation for selecting a subset of \f$ k \f$ columns.
     * @param X The input matrix (dimensions \f$ m \times n \f$) from which
     * columns are to be selected. It is assumed that \f$ X \f$ is full rank
     * for theoretical guarantees.
     * @param k The number of columns to select. Must satisfy \f$ k = m \f$.
     * @param swap_count Unused; Q-DEIM performs no exchanges and leaves the
     * caller's -1 ("not tracked") in place.
     * @return A `std::vector` of `Eigen::Index` containing the 0-based indices
     * of the selected columns.
     */
    std::vector<Eigen::Index>
    selectSubsetImpl(const Eigen::MatrixX<Scalar> &X, Eigen::Index k,
                     [[maybe_unused]] Eigen::Index *swap_count) override {

        const Eigen::Index m = X.rows();

        assert(k == m && "QdeimSelector: Q-DEIM returns one interpolation "
                         "point per basis vector, so only k = m is supported.");

        // Make a copy to permute in-place.
        Eigen::MatrixX<Scalar> R = X;

        // The CPQR strategy stops right after the m pivots are chosen, with no
        // greedy or exchange phase, which is precisely Q-DEIM.
        std::vector<Eigen::Index> indices =
            VolumePivotingBase<Scalar>::selectStartingSet(
                R, k, Initialization::CPQR);

        indices.resize(static_cast<size_t>(k));
        return indices;
    }
};

} // namespace MatSubset

#endif // MAT_SUBSET_QDEIM_SELECTOR_H
