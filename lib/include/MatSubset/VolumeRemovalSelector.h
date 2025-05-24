#ifndef MAT_SUBSET_VOLUME_REMOVAL_SELECTOR_H
#define MAT_SUBSET_VOLUME_REMOVAL_SELECTOR_H

#include <Eigen/SVD> // For Eigen::BDCSVD

#include "SelectorBase.h" // Base class

namespace MatSubset {

/*!
 * @brief Class for approximating subset selection problem for matrices using
 * Volume-based greedy removal strategy.
 * @tparam Scalar The underlying scalar type (e.g., `float`, `double`).
 *
 * This class implements a simplified version of algorithm used in
 * `FrobeniusRemovalSelector`. Removing numerator from formulas used in
 * mentioned algorithm produces a new one, where the removed column guarantees
 * the maximum possible volume (product of singular values) of the remaining
 * submatrix.
 *
 */
template <typename Scalar>
class VolumeRemovalSelector : public SelectorBase<Scalar> {
  public:
    /*!
     * @brief Default constructor for `VolumeRemovalSelector`.
     */
    VolumeRemovalSelector() = default;

    /*!
     * @brief Gets the human-readable name of the algorithm.
     * @return The string "volume removal".
     */
    std::string getAlgorithmName() const override { return "volume removal"; }

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
        const Eigen::Index num_cols_to_remove = n_initial_cols - k;

        std::vector<Eigen::Index> current_col_indices(n_initial_cols);
        for (Eigen::Index j = 0; j < n_initial_cols; ++j) {
            current_col_indices[static_cast<size_t>(j)] = j;
        }

        Eigen::BDCSVD<Eigen::MatrixX<Scalar>> svd(X, Eigen::ComputeThinV);
        Eigen::MatrixX<Scalar> V_matrix = svd.matrixV().transpose();

        Eigen::MatrixX<Scalar> V_dag =
            (V_matrix * V_matrix.transpose()).inverse() * V_matrix;
        Eigen::ArrayX<Scalar> d_scores =
            static_cast<Scalar>(1.0) -
            (V_matrix.transpose() * V_dag).diagonal().array();

        for (Eigen::Index iter = 0; iter < num_cols_to_remove; ++iter) {

            Eigen::Index j_max_idx;
            d_scores.maxCoeff(&j_max_idx);

            Eigen::VectorX<Scalar> w_V = V_matrix.col(j_max_idx);
            Eigen::VectorX<Scalar> w_V_dag = V_dag.col(j_max_idx);
            Scalar d_max_val_removed = d_scores(j_max_idx);

            removeColumn(current_col_indices, d_scores, V_matrix, V_dag,
                         j_max_idx);

            d_scores -=
                (w_V.transpose() * V_dag).array().square() / d_max_val_removed;
            V_dag +=
                w_V_dag * (w_V_dag.transpose() * V_matrix) / d_max_val_removed;
        }

        return current_col_indices;
    }

  private:
    /*!
     * @brief Helper to remove column `idx_to_remove` from active data
     * structures.
     * @param col_indices Vector of original column indices.
     * @param d_scores Array of d-scores.
     * @param V Matrix of active V columns.
     * @param V_dag Matrix of active V_dag columns.
     * @param idx_to_remove The 0-based index *within the current active set* to
     * remove.
     */
    void removeColumn(std::vector<Eigen::Index> &col_indices,
                      Eigen::ArrayX<Scalar> d_scores, Eigen::MatrixX<Scalar> V,
                      Eigen::MatrixX<Scalar> V_dag,
                      Eigen::Index idx_to_remove) {

        Eigen::Index new_size =
            static_cast<Eigen::Index>(col_indices.size()) - 1;

        if (idx_to_remove < new_size) {
            col_indices[static_cast<size_t>(idx_to_remove)] =
                col_indices[static_cast<size_t>(new_size)];
            d_scores(idx_to_remove) = d_scores(new_size);
            V.col(idx_to_remove) = V.col(new_size);
            V_dag.col(idx_to_remove) = V_dag.col(new_size);
        }

        col_indices.resize(static_cast<size_t>(new_size));
        d_scores.conservativeResize(new_size);
        V.conservativeResize(Eigen::NoChange, new_size);
        V_dag.conservativeResize(Eigen::NoChange, new_size);
    }
};

} // namespace MatSubset

#endif // MAT_SUBSET_VOLUME_REMOVAL_SELECTOR_H