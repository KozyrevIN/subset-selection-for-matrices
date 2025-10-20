#ifndef MAT_SUBSET_DOMINANT_SELECTOR_H
#define MAT_SUBSET_DOMINANT_SELECTOR_H

#include <cassert>  // For assert
#include <cmath>    // For std::log, std::ceil
#include <iostream> // For std::cerr

#include <Eigen/QR> // For Eigen::CompleteOrthogonalDecomposition

#include "ColumnPivotingSelector.h" // Base class
#include "Enums.h"                  // For MatSubset::Norm

namespace MatSubset {

/*!
 * @brief Approximates subset selection problem for matrices by finding
 * c-locally optimum volume submatrix.
 * @tparam Scalar The underlying scalar type (e.g., `float`, `double`).
 *
 * This class implements Algorithm 3 from Osinsky
 * (2024), "Volume-based Subset Selection". For \f$ k = m \f$ this algorithm is
 * equivalent to Maxvol algorithm (Goreinov et. al. (2010), "How to find a good
 * submatrix") and to selecting pivot columns identified by strong RRQR
 * algorithm (Gu and Eisenstat (1996), "Efficient algorithms for computing a
 * strong rank-revealing qr factorization")
 *
 * The algorithm selects the initial subset with nonzero volume by running
 * rank-revealing QR and adding first \f$ k-m \f$ non-selected columns. It then
 * iteratively refines tis set \f$ S \f$ by finding sunch indices \f$ i \in
 * S \f$ and \f$ j \notin S \f$, that replacing i with j increases the squared
 * volume of \f$ X_S \f$ by \f$ c \f$ times. If no such repacement exists,
 * algorithm terminates.
 */
template <typename Scalar>
class DominantSelector : public ColumnPivotingSelector<Scalar> {
  public:
    /*!
     * @brief Default constructor for `DominantSelector`.
     */
    DominantSelector(Scalar c) : c(c) {
        assert(c > 1 &&
               "In the dominant algorithm parameter c must be greater then 1.");
    };

    /*!
     * @brief Gets the human-readable name of the algorithm.
     * @return The string "dominant".
     */
    std::string getAlgorithmName() const override { return "dominant"; }

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

        // Preparing starting sets of indices
        std::vector<Eigen::Index> selected_indices =
            ColumnPivotingSelector<Scalar>::selectSubsetImpl(X, m);
        std::vector<Eigen::Index> remaining_indices;
        remaining_indices.reserve(n - k);

        std::vector<bool> is_already_selected(n, false);
        for (Eigen::Index i : selected_indices) {
            is_already_selected[static_cast<size_t>(i)] = true;
        }

        for (Eigen::Index i = 0; i < n; ++i) {
            if (!is_already_selected[static_cast<size_t>(i)]) {
                if (selected_indices.size() < k) {
                    selected_indices.push_back(i);
                } else {
                    remaining_indices.push_back(i);
                }
            }
        }

        // Initializing neccesary matrices
        Eigen::MatrixX<Scalar> R(m, n);
        R.leftCols(k) = X(Eigen::all, selected_indices);
        R.rightCols(n - k) = X(Eigen::all, remaining_indices);

        Eigen::MatrixX<Scalar> R_selected_dag =
            R.leftCols(k).completeOrthogonalDecomposition().pseudoInverse();

        Eigen::MatrixX<Scalar> C = R_selected_dag * R;
        auto C_selected = C.leftCols(k);
        auto C_remaining = C.rightCols(n - k);

        Eigen::ArrayX<Scalar> l = C.colwise().squaredNorm();
        auto l_selected = l.head(k);
        auto l_remaining = l.tail(n - k);

        Eigen::MatrixXf::Index i_max, j_max;
        Scalar max_val;
        Eigen::MatrixX<Scalar> B;
        if (k > m) {
            B = (1 - l_selected).matrix() *
                    (1 + l_remaining).matrix().transpose() +
                C_remaining.array().abs2().matrix();

            max_val = B.maxCoeff(&i_max, &j_max);
        } else {
            max_val = C_remaining.cwiseAbs().maxCoeff(&i_max, &j_max);
            max_val *= max_val;
        }

        // Computing maximun possible number of swaps
        Eigen::Index max_swap_count = static_cast<Eigen::Index>(
            std::ceil(2 * m * std::log(k) / std::log(c)));
        Eigen::Index swap_count = 0;

        // Main loop
        Eigen::MatrixX<Scalar> last_row(1, n);
        while ((max_val > c) && (swap_count <= max_swap_count)) {

            // Add extra column
            last_row = C_remaining.col(j_max).transpose() * C /
                       (1 + l_remaining(j_max));
            C -= C_remaining.col(j_max) * last_row;
            l -= last_row.transpose().array().abs2() / (1 + l_remaining(j_max));

            // Swap newly added column and one destined to removal
            std::swap(selected_indices[static_cast<size_t>(i_max)],
                      remaining_indices[static_cast<size_t>(j_max)]);
            std::swap(l_selected(i_max), l_remaining(j_max));

            C_selected.col(i_max).swap(C_remaining.col(j_max));
            std::swap(last_row(i_max), last_row(k + j_max));
            C.row(i_max).swap(last_row);
            swap_count++;

            // Remove the column
            l += last_row.transpose().array().abs2() / (1 - l_remaining(j_max));
            C += C_remaining.col(j_max) * last_row * (1 + l_remaining(j_max));

            // Select indices to swap
            if (k > m) {
                B = (1 - l_selected).matrix() *
                        (1 + l_remaining).matrix().transpose() +
                    C_remaining.array().abs2().matrix();

                max_val = B.maxCoeff(&i_max, &j_max);
            } else {
                max_val = C_remaining.cwiseAbs().maxCoeff(&i_max, &j_max);
                max_val *= max_val;
            }
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

        return selected_indices;
    }

    /*!
     * @brief Calculates the theoretical bound for the dominant selection
     * strategy.
     * @param m The number of rows in the matrix.
     * @param n The number of columns in the matrix.
     * @param k The number of columns that would be selected.
     * @param norm The type of matrix norm (`Norm::Frobenius` or
     * `Norm::Spectral`).
     * @return A `Scalar` value representing the calculated lower bound on the
     * ratio \f$ \lVert X^{\dag} \rVert^{2}/\lVert X_{\mathcal{S}}^{\dag}
     * \rVert^{2} \f$.
     *
     * The bound is calculated based on the Corollary 1 in Osinsky (2024).
     *
     * @note The bound depends on the parameter c.
     * @note The bound for Frobenius norm from Osinsky (2024),
     * Corollary 1, is not standard (it mixes different norms), we use
     * inequality \f$ \lVert X \rVert_2 \le \lVert X \rVert_F \f$ to produce a
     * looser bound, wich fits into our framework.
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
     * @brief Parameter determining the algorithms stopping criterion.
     *
     * Algorithms stops when finds \f$ \sqrt{c} \f$ locally maximum volume
     * submatrix.
     */
    Scalar c;
};

} // namespace MatSubset

#endif // MAT_SUBSET_DOMINANT_SELECTOR_H