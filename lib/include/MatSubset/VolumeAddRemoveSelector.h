#ifndef MAT_SUBSET_VOLUME_ADD_REMOVE_SELECTOR_H
#define MAT_SUBSET_VOLUME_ADD_REMOVE_SELECTOR_H

#include <cassert>  // For assert
#include <cmath>    // For std::log, std::ceil
#include <iostream> // For std::cerr

#include <Eigen/LU> // For .inverse()
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
     */
    explicit VolumeAddRemoveSelector(Scalar c) : c(c) {
        assert(c > 1 &&
               "In the volume add-remove algorithm parameter c must be "
               "greater than 1.");
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
    std::vector<Eigen::Index> selectSubsetImpl(const Eigen::MatrixX<Scalar> &X,
                                               Eigen::Index k) override {

        const Eigen::Index m = X.rows();
        const Eigen::Index n = X.cols();

        // Make a copy to permute in-place
        Eigen::MatrixX<Scalar> R = X;

        // Permute columns so first m columns form a high-volume submatrix
        std::vector<Eigen::Index> indices =
            VolumePivotingBase<Scalar>::selectStartingSet(R);

        // Initialize necessary matrices
        Eigen::MatrixX<Scalar> Y = Eigen::MatrixX<Scalar>::Identity(m, m);
        Eigen::HouseholderQR<Eigen::MatrixX<Scalar>> qr(
            R.leftCols(k).transpose());
        Eigen::MatrixX<Scalar> L = qr.matrixQR()
                                       .topRows(m)
                                       .template triangularView<Eigen::Upper>()
                                       .transpose();
        Eigen::MatrixX<Scalar> C =
            L.template triangularView<Eigen::Lower>().solve(R);
        auto C_selected = C.leftCols(k);
        auto C_remaining = C.rightCols(n - k);

        // And arrays
        Eigen::ArrayX<Scalar> l = C.colwise().squaredNorm();
        auto l_selected = l.head(k);
        auto l_remaining = l.tail(n - k);
        Eigen::Index j_max;
        Scalar l_j_max = l_remaining.maxCoeff(&j_max);
        Eigen::VectorX<Scalar> v = C_remaining.col(j_max);
        Eigen::ArrayX<Scalar> term_1 =
            (v.transpose() * C_selected).array().square();
        Eigen::ArrayX<Scalar> b = term_1 + (1 - l_selected) * (1 + l_j_max);
        Eigen::Index i_max;

        // DEBUG: Verify l after column addition
        {
            Eigen::MatrixX<Scalar> R_selected_with_j = R.leftCols(k);
            Eigen::MatrixX<Scalar> R_selected_pinv_R =
                R_selected_with_j.completeOrthogonalDecomposition()
                    .pseudoInverse() *
                R;
            Eigen::ArrayX<Scalar> l_expected =
                R_selected_pinv_R.colwise().squaredNorm();
            Scalar max_diff = (l - l_expected).abs().maxCoeff();
            std::cerr << "Before swaps: max_diff = " << max_diff << std::endl;
        }

        // Computing maximum possible number of swaps
        Eigen::Index max_swap_count = static_cast<Eigen::Index>(
            std::ceil(2 * m * std::log(k) / std::log(c)));
        Eigen::Index swap_count = 0;

        // Main loop
        Eigen::ArrayX<Scalar> extra_row;
        while ((b.maxCoeff(&i_max) > c) && (swap_count <= max_swap_count)) {
            v = C_remaining.col(j_max);
            Eigen::VectorX<Scalar> v_2 = Y * v;
            Y -= v_2 * v_2.transpose() * Y / (1 + l_j_max);
            l = ((C_selected.transpose() * Y) * C).colwise().squaredNorm();

            // DEBUG: Verify l after column addition
            {
                Eigen::MatrixX<Scalar> R_selected_with_j = R.leftCols(k + 1);
                R_selected_with_j.col(k) = R.col(k + j_max);
                Eigen::MatrixX<Scalar> R_selected_pinv_R =
                    R_selected_with_j.completeOrthogonalDecomposition()
                        .pseudoInverse() *
                    R;
                Eigen::ArrayX<Scalar> l_expected =
                    R_selected_pinv_R.colwise().squaredNorm();
                Scalar max_diff = (l - l_expected).abs().maxCoeff();
                std::cerr << "After addition (swap " << swap_count
                          << "): max_diff = " << max_diff << std::endl;
                l = l_expected;
            }

             // Swap newly added column and one destined to removal
            std::swap(indices[static_cast<size_t>(i_max)],
                      indices[static_cast<size_t>(k + j_max)]);
            std::swap(l_selected(i_max), l_remaining(j_max));
            C_selected.col(i_max).swap(C_remaining.col(j_max));
            swap_count++;

            v = C_remaining.col(j_max);
            v_2 = Y * v;
            Y += v_2 * v_2.transpose() / (1 - l_remaining(j_max));
            l = ((C.transpose() * Y) * C).diagonal().array().square();

            {
                // DEBUG: Verify l after column removal

                Eigen::MatrixX<Scalar> R_selected_pinv_R =
                    R.leftCols(k)
                        .completeOrthogonalDecomposition()
                        .pseudoInverse() *
                    R;
                Eigen::ArrayX<Scalar> l_expected =
                    R_selected_pinv_R.colwise().squaredNorm();
                Scalar max_diff = (l - l_expected).abs().maxCoeff();
                std::cerr << "After removal (swap " << swap_count
                          << "): max_diff = " << max_diff << std::endl
                          << std::endl;
            }

            /*
            // Recalculate l and Y upon column addition
            v = C_remaining.col(j_max);
            Eigen::VectorX<Scalar> v_2 = Y * v;
            Y -= v_2 * v_2.transpose() * Y / (1 + l_j_max);
            extra_row = ((v.transpose() * Y) * C).array().square();
            l -= extra_row * (1 + l_j_max);

            // DEBUG: Verify l after column addition
            {
                Eigen::MatrixX<Scalar> R_selected_with_j = R.leftCols(k + 1);
                R_selected_with_j.col(k) = R.col(k + j_max);
                Eigen::MatrixX<Scalar> R_selected_pinv_R =
                    R_selected_with_j.completeOrthogonalDecomposition()
                        .pseudoInverse() *
                    R;
                Eigen::ArrayX<Scalar> l_expected =
                    R_selected_pinv_R.colwise().squaredNorm();
                Scalar max_diff = (l - l_expected).abs().maxCoeff();
                std::cerr << "After addition (swap " << swap_count
                          << "): max_diff = " << max_diff << std::endl;
                l = l_expected;
            }

            // Swap newly added column and one destined to removal
            std::swap(indices[static_cast<size_t>(i_max)],
                      indices[static_cast<size_t>(k + j_max)]);
            std::swap(l_selected(i_max), l_remaining(j_max));
            C_selected.col(i_max).swap(C_remaining.col(j_max));
            swap_count++;

            // Recalculate l and Y upon column removal
            v = C_remaining.col(j_max);
            v_2 = Y * v;
            Y += v_2 * v_2.transpose() / (1 - l_remaining(j_max));
            extra_row = ((v.transpose() * Y) * C).array().square();
            l += extra_row * (1 - l_remaining(j_max));

            // DEBUG: Verify l after column removal

            Eigen::MatrixX<Scalar> R_selected_pinv_R =
                R.leftCols(k)
                    .completeOrthogonalDecomposition()
                    .pseudoInverse() *
                R;
            Eigen::ArrayX<Scalar> l_expected =
                R_selected_pinv_R.colwise().squaredNorm();
            Scalar max_diff = (l - l_expected).abs().maxCoeff();
            std::cerr << "After removal (swap " << swap_count
                      << "): max_diff = " << max_diff << std::endl
                      << std::endl;
            l = l_expected;
            */

            // Find new indices for replacement
            l_j_max = l_remaining.maxCoeff(&j_max);
            v = C_remaining.col(j_max);
            term_1 = ((v.transpose() * Y) * C_selected).array().square();
            // term_1 =
            // R_selected_pinv_R.col(j_max).transpose().array().square();
            b = term_1 + (1 - l_selected) * (1 + l_j_max);
        }

        // Warning if maximum swap count was reached
        if (swap_count > max_swap_count) {
            std::cerr
                << "Warning: VolumeAddRemoveSelector reached maximum swap "
                   "count ("
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
};

} // namespace MatSubset

#endif // MAT_SUBSET_VOLUME_ADD_REMOVE_SELECTOR_H
