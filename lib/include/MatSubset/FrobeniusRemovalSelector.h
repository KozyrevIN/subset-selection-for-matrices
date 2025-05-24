#ifndef MAT_SUBSET_FROBENIUS_REMOVAL_SELECTOR_H
#define MAT_SUBSET_FROBENIUS_REMOVAL_SELECTOR_H

#include <Eigen/SVD> // For Eigen::BDCSVD

#include "SelectorBase.h" // Base class

namespace MatSubset {

/*!
 * @brief Class for approximating subset selection problem for matrices using
 * Frobenius norm-based greedy removal strategy.
 * @tparam Scalar The underlying scalar type (e.g., `float`, `double`).
 *
 * This class implements a modification of Algorithm 1 ("Deterministic Greedy
 * Removal (Frobenius norm)") from Avron and Boutsidis (2012), "Faster Subset
 * Selection for Matrices and Applications".
 *
 * In each step, it greedily removes the column from the currently active set,
 * to maximize the Frobenius norm of remaining submatrix. The key difference
 * from the original algorithm from the article is that our modification does
 * not require recalculation of SVD on each step.
 *
 * The \f$ \epsilon \f$ parameter is used as a threshold for values in
 * denominator, to ensure numerical stability.
 */
template <typename Scalar>
class FrobeniusRemovalSelector : public SelectorBase<Scalar> {
  public:
    /*!
     * @brief Constructor for `FrobeniusRemovalSelector`.
     * @param eps Small positive tolerance value. Defaults to \f$ 1e-6 \f$.
     */
    explicit FrobeniusRemovalSelector(Scalar eps = static_cast<Scalar>(1e-6))
        : eps_(eps) {}

    /*!
     * @brief Gets the human-readable name of the algorithm.
     * @return The string "frobenius removal".
     */
    std::string getAlgorithmName() const override {
        return "frobenius removal";
    }

  protected:
    /*!
     * @brief Core implementation for selecting a subset of \f$ k \f$ columns.
     * @param X The \f$ m \times n \f$ input matrix \f$ X \f$.
     * @param k The number of columns to select.
     * @return A `std::vector` of `Eigen::Index` of selected column indices.
     */
    std::vector<Eigen::Index> selectSubsetImpl(const Eigen::MatrixX<Scalar> &X,
                                               Eigen::Index k) override {

        const Eigen::Index n_initial_cols = X.cols();

        std::vector<Eigen::Index> current_col_indices(n_initial_cols);
        for (Eigen::Index j = 0; j < n_initial_cols; ++j) {
            current_col_indices[static_cast<size_t>(j)] = j;
        }

        Eigen::BDCSVD<Eigen::MatrixX<Scalar>> svd(X, Eigen::ComputeThinV);
        Eigen::MatrixX<Scalar> V = svd.matrixV().transpose();
        const Eigen::VectorX<Scalar> S_inv2 =
            svd.singularValues().array().inverse().square();

        // V_dag is a pseudoinverse of V
        Eigen::MatrixX<Scalar> V_dag = (V * V.transpose()).inverse() * V;

        // l_scores and d_scores are the numerators and denominators,
        // respectively, for the score function l_j / d_j, which is minimized to
        // select a column for removal. They are initialized based on X and then
        // updated in each iteration.
        Eigen::ArrayX<Scalar> l_scores =
            (V_dag.transpose() * S_inv2.asDiagonal() * V_dag).diagonal();
        Eigen::ArrayX<Scalar> d_scores =
            static_cast<Scalar>(1.0) -
            (V.transpose() * V_dag).diagonal().array();

        // main loop in which removal of columns happens.
        // `n` here refers to the `n` from the base class context (X.cols())
        for (Eigen::Index active_size = X.cols(); active_size > k; --active_size) {

            Eigen::Index j_min_idx = 0; // Index within the *active* set
            // Find first valid candidate for j_min_idx
            bool found_initial_j_min = false;
            for (Eigen::Index j_check = 0; j_check < active_size; ++j_check) {
                if (d_scores(j_check) > eps_) {
                    j_min_idx = j_check;
                    found_initial_j_min = true;
                    break;
                }
            }
            assert(found_initial_j_min && "No suitable column found for removal; all d_scores <= eps_.");


            for (Eigen::Index j = 0; j < active_size; ++j) {
                if (d_scores(j) > eps_ && (l_scores(j) * d_scores(j_min_idx) <
                                           l_scores(j_min_idx) * d_scores(j))) {
                    j_min_idx = j;
                }
            }

            Eigen::VectorX<Scalar> w_V = V.col(j_min_idx);
            Eigen::VectorX<Scalar> w_V_dag = V_dag.col(j_min_idx);
            Scalar d_min_val = d_scores(j_min_idx);

            // removeColumn effectively swaps element j_min_idx with the last
            // active element, then resizes all active data structures.
            removeColumn(current_col_indices, l_scores, d_scores, V, V_dag,
                         j_min_idx);
            // After removeColumn, current_col_indices.size() is active_size - 1

            // V, V_dag, l_scores, d_scores are now one element/column
            // smaller. Their new size is current_col_indices.size().

            // Update V_dag:
            V_dag += w_V_dag * (w_V_dag.transpose() * V) / d_min_val;

            // Update l_scores and d_scores for remaining active columns:
            Eigen::ArrayX<Scalar> mul_1 = (w_V.transpose() * V_dag).array();
            Eigen::ArrayX<Scalar> mul_2 =
                (w_V_dag.transpose() * S_inv2.asDiagonal() * V_dag).array();

            Scalar l_update_scalar_term = mul_2(
                static_cast<Eigen::Index>(current_col_indices.size()) - 1);

            l_scores += (static_cast<Scalar>(2.0) * mul_1 * mul_2 / d_min_val) +
                        (mul_1.square() * l_update_scalar_term /
                         (d_min_val * d_min_val));

            d_scores -= mul_1.square() / d_min_val;
        }

        return current_col_indices;
    }

    /*!
     * @brief Calculates the theoretical bound.
     * @param m Number of rows in the original matrix (\f$ m \f$).
     * @param n Number of columns in the original matrix (\f$ n \f$).
     * @param k Number of selected columns (\f$ k \f$).
     * @param norm The norm type (`Norm::Frobenius` or `Norm::Spectral`).
     * @return The calculated bound based on Theorem 3.1 of Avron & Boutsidis
     * (2012).
     */
    Scalar boundImpl(Eigen::Index m, Eigen::Index n, Eigen::Index k,
                     Norm norm) const override {
        // Preconditions (m, n, k >=1, m <= k <= n) are handled by SelectorBase.
        // This ensures n - m + 1 >= 1.
        Scalar bound_val =
            static_cast<Scalar>(k - m + 1) / static_cast<Scalar>(n - m + 1);
        if (norm == Norm::Spectral) {
            bound_val /= static_cast<Scalar>(m);
        }
        return bound_val;
    }

  private:
    Scalar eps_; /*!< Tolerance for \f$ d_j \f$ values. */

    /*!
     * @brief Helper to remove column `idx_to_remove` from active data
     * structures. Modifies all parameters in place by copying the last active
     * element to `idx_to_remove` and then resizing.
     * @param col_indices Vector of original column indices.
     * @param l_scores Array of l-scores (numerators for the removal criterion).
     * @param d_scores Array of d-scores (denominators for the removal criterion).
     * @param V Matrix of active \f$ V \f$ columns (from SVD).
     * @param V_dag Matrix of active \f$ V^{\dag} \f$ (pseudoinverse related) columns.
     * @param idx_to_remove The 0-based index *within the current active set* to
     * remove.
     */
    void removeColumn(std::vector<Eigen::Index> &col_indices,
                      Eigen::ArrayX<Scalar> &l_scores,
                      Eigen::ArrayX<Scalar> &d_scores,
                      Eigen::MatrixX<Scalar> &V, Eigen::MatrixX<Scalar> &V_dag,
                      Eigen::Index idx_to_remove) {

        // `col_indices` is the vector of original indices, its size is the current active_size.
        Eigen::Index new_size = static_cast<Eigen::Index>(col_indices.size()) - 1;
        // new_size will be >= k (target selection size) >= m (rows) >= 1.

        if (idx_to_remove < new_size) {
            col_indices[static_cast<size_t>(idx_to_remove)] =
                col_indices[static_cast<size_t>(new_size)];
            l_scores(idx_to_remove) = l_scores(new_size);
            d_scores(idx_to_remove) = d_scores(new_size);
            V.col(idx_to_remove) = V.col(new_size);
            V_dag.col(idx_to_remove) = V_dag.col(new_size);
        }

        col_indices.resize(static_cast<size_t>(new_size));
        l_scores.conservativeResize(new_size);
        d_scores.conservativeResize(new_size);
        V.conservativeResize(Eigen::NoChange, new_size);
        V_dag.conservativeResize(Eigen::NoChange, new_size);
    }
};

} // namespace MatSubset

#endif // MAT_SUBSET_FROBENIUS_REMOVAL_SELECTOR_H