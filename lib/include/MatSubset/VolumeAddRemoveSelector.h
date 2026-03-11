#ifndef MAT_SUBSET_VOLUME_ADD_REMOVE_SELECTOR_H
#define MAT_SUBSET_VOLUME_ADD_REMOVE_SELECTOR_H

#include <cassert>  // For assert
#include <cmath>    // For std::log, std::ceil
#include <iostream> // For std::cerr (warnings)

#include <Eigen/QR> // For Eigen::HouseholderQR

#include "Enums.h"              // For MatSubset::Norm
#include "VolumePivotingBase.h" // Base class

namespace MatSubset {

/*!
 * @brief Approximates subset selection problem for matrices using volume-based
 * addition followed by removal strategy.
 * @tparam Scalar The underlying scalar type (e.g., `float`, `double`).
 *
 * This class implements a variant of the Dominant algorithm that achieves the
 * same theoretical bounds with reduced complexity. Instead of searching for
 * optimal replacement (simultaneous add-remove), this algorithm first finds
 * the optimal column to add, then finds the optimal column to remove.
 *
 * The algorithm selects the initial subset with nonzero volume by running
 * `VolumePivotingBase` routine and adding first \f$ k-m \f$ non-selected
 * columns. It then iteratively refines this set \f$ S \f$ by:
 * 1. Finding column \f$ j \notin S \f$ that maximally increases volume when
 * added
 * 2. Finding column \f$ i \in S \f$ that minimally decreases volume when
 * removed If the net volume increase is at least \f$ \sqrt{c} \f$ times, the
 * swap is performed. Algorithm terminates when no such improvement exists.
 */
template <typename Scalar>
class VolumeAddRemoveSelector : public VolumePivotingBase<Scalar> {
  public:
    /*!
     * @brief Constructor for `VolumeAddRemoveSelector`.
     * @param c The improvement threshold parameter. Must be greater than 1.
     * Algorithm stops when no add-remove pair improves squared volume by factor
     * c.
     * @param greedy_init If true, use greedy selection/removal in
     * `selectStartingSet` to initialize k columns instead of just m.
     * @param oversampling The number of extra columns to greedily add before
     * removing back down to k during initialization. Only used when
     * `greedy_init` is true.
     */
    explicit VolumeAddRemoveSelector(Scalar c, bool greedy_init = false,
                                     Eigen::Index oversampling = 0)
        : c(c), greedy_init(greedy_init), oversampling(oversampling) {
        assert(c > 1 &&
               "In the volume add-remove algorithm parameter c must be "
               "greater than 1.");
        assert((greedy_init || (oversampling == 0)) &&
               "oversampling must be 0 if greedy init is disabled");
    };

    /*!
     * @brief Gets the human-readable name of the algorithm.
     * @return The string "volume add-remove".
     */
    std::string getAlgorithmName() const override {
        return "volume add-remove";
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
    selectSubsetImpl(const Eigen::MatrixX<Scalar> &X,
                     Eigen::Index k) override {

        const Eigen::Index m = X.rows();
        const Eigen::Index n = X.cols();

        // Make a copy to permute in-place
        Eigen::MatrixX<Scalar> R = X;

        // Permute columns so first m (or k) columns form a high-volume submatrix
        std::vector<Eigen::Index> indices =
            VolumePivotingBase<Scalar>::selectStartingSet(
                R, greedy_init ? k : m, oversampling);

        // Initialize l and Y
        Eigen::MatrixX<Scalar> Y =
            (R.leftCols(k) * R.leftCols(k).transpose()).inverse();
        Eigen::VectorX<Scalar> l = (X.transpose() * Y * X).diagonal();

        // Compute maximum possible number of swaps
        Eigen::Index max_swap_count = std::numeric_limits<Eigen::Index>::max();
        if (c > 1) {
            max_swap_count = static_cast<Eigen::Index>(
                std::ceil(2 * m * std::log(k) / std::log(c)));
        }
        Eigen::Index swap_count = 0;

        // Main loop with column exchange
        while (swap_count < max_swap_count) {
            // Column selection
            Eigen::Index s;
            Scalar l_s = l.tail(n - k).maxCoeff(&s);
            Eigen::VectorX<Scalar> Y_r_s = Y * R.col(s);
            l -= (Y_r_s.transpose() * R).cwiseAbs2() / (1 + l_s);
            Y -= Y_r_s * Y_r_s.transpose() / (1 + l_s);

            // Column removal
            Eigen::Index r;
            Scalar l_r = l.head(k).minCoeff(&r);
            if ((1 + l_s) * (1 - l_r) <= c) {
                break;
            }
            Eigen::VectorX<Scalar> Y_r_r = Y * R.col(r);
            l += (Y_r_r.transpose() * R).cwiseAbs2() / (1 + l_r);
            Y -= Y_r_r * Y_r_r.transpose() / (1 + l_r);

            // Update
            std::swap(indices[static_cast<size_t>(r)],
                      indices[static_cast<size_t>(s)]);
            std::swap(l(r), l(s));
            R.col(r).swap(R.col(s));
            swap_count++;
        }

        // Warning if maximum swap count was reached
        if (swap_count > max_swap_count) {
            std::cerr
                << "Warning: DominantSelector reached maximum swap count ("
                << max_swap_count
                << "). This is theoretically impossible and may indicate "
                   "numerical errors or invalid input matrix (e.g., "
                   "rank-deficient)."
                << std::endl;
        }

        indices.resize(k);
        return indices;
    }

    /*!
     * @brief Calculates the theoretical bound for the volume add-remove
     * selection strategy.
     * @param m The number of rows in the matrix.
     * @param n The number of columns in the matrix.
     * @param k The number of columns that would be selected.
     * @param norm The type of matrix norm (`Norm::Frobenius` or
     * `Norm::Spectral`).
     * @return A `Scalar` value representing the calculated lower bound on the
     * ratio \f$ \lVert X^{\dag} \rVert^{2}/\lVert X_{\mathcal{S}}^{\dag}
     * \rVert^{2} \f$.
     *
     * The bound is identical to DominantSelector, as both algorithms achieve
     * the same theoretical guarantees.
     *
     * @note The bound depends on the parameter c.
     * @note The bound for Frobenius norm is not standard (it mixes different
     * norms), we use inequality \f$ \lVert X \rVert_2 \le \lVert X \rVert_F \f$
     * to produce a looser bound, which fits into our framework.
     */
    Scalar boundImpl(Eigen::Index m, Eigen::Index n, Eigen::Index k,
                     Norm norm) const override {

        Scalar sum_k_m = static_cast<Scalar>(m + (c - 1) * k);
        Scalar diff_n_k = static_cast<Scalar>(n - k);
        Scalar diff_k_m = static_cast<Scalar>(k - m + 1);
        Scalar core_mult = sum_k_m * diff_n_k / diff_k_m;
        Scalar S_1 = static_cast<Scalar>(1);

        if (norm == Norm::Spectral) {
            return S_1 / (S_1 + core_mult);
        } else {
            Scalar S_m = static_cast<Scalar>(m);
            return S_1 / (S_m + core_mult);
        }
    }

  private:
    /*!
     * @brief Parameter determining the algorithm's stopping criterion.
     *
     * Algorithm stops when no add-remove pair improves squared volume by
     * factor \f$ c \f$.
     */
    Scalar c;

    /*!
     * @brief If true, use greedy selection/removal to initialize k columns
     * instead of just m.
     */
    bool greedy_init;

    /*!
     * @brief Number of extra columns to greedily add before removing back
     * down to k during initialization.
     */
    Eigen::Index oversampling;
};

} // namespace MatSubset

#endif // MAT_SUBSET_VOLUME_ADD_REMOVE_SELECTOR_H
