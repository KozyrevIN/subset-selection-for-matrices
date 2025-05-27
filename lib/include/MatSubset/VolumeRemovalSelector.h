#ifndef MAT_SUBSET_VOLUME_REMOVAL_SELECTOR_H
#define MAT_SUBSET_VOLUME_REMOVAL_SELECTOR_H

#include <Eigen/QR>  // For completeOrthogonalDecomposition
#include <Eigen/SVD> // For Eigen::BDCSVD

#include "SelectorBase.h" // Base class

namespace MatSubset {

/*!
 * @brief Approximates subset selection problem for matrices using
 * volume-based greedy removal strategy.
 * @tparam Scalar The underlying scalar type (e.g., `float`, `double`).
 *
 * This class implements a simplified version of algorithm used in
 * `FrobeniusRemovalSelector`. Removing numerator from formulas used in
 * mentioned algorithm produces a new one, where the removed column guarantees
 * the maximum possible volume (product of singular values) of the remaining
 * submatrix.
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

        std::vector<Eigen::Index> cols(n);
        for (Eigen::Index j = 0; j < n; ++j) {
            cols[j] = j;
        }

        Eigen::JacobiSVD<Eigen::MatrixX<Scalar>> svd(X, Eigen::ComputeThinV);
        Eigen::MatrixX<Scalar> V = svd.matrixV().transpose();

        Eigen::MatrixX<Scalar> V_dag =
            V.completeOrthogonalDecomposition().pseudoInverse().transpose();
        Eigen::ArrayX<Scalar> d =
            1 - (V.transpose() * V_dag).diagonal().array();

        while (cols.size() > k) {
            Eigen::Index j_max;
            Scalar d_max = d.maxCoeff(&j_max);

            Eigen::VectorX<Scalar> w = V.col(j_max);
            Eigen::VectorX<Scalar> w_dag = V_dag.col(j_max);

            removeColumn(cols, d, V, V_dag, j_max);

            d -= (w.transpose() * V_dag).array().square() / d_max;
            V_dag += w_dag * (w_dag.transpose() * V) / d_max;
        }

        return cols;
    }

  private:
    /*!
     * @brief Helper to remove column `idx_to_remove` from active data
     * structures.
     * @param cols Vector of original column indices.
     * @param d Array of d-scores.
     * @param V Matrix of active V columns.
     * @param V_dag Matrix of active V_dag columns.
     * @param idx_to_remove The 0-based index *within the current active set* to
     * remove.
     */
    void removeColumn(std::vector<Eigen::Index> &cols, Eigen::ArrayX<Scalar> d,
                      Eigen::MatrixX<Scalar> V, Eigen::MatrixX<Scalar> V_dag,
                      Eigen::Index idx_to_remove) {

        Eigen::Index new_size = static_cast<Eigen::Index>(cols.size()) - 1;

        if (idx_to_remove < new_size) {
            cols[static_cast<size_t>(idx_to_remove)] =
                cols[static_cast<size_t>(new_size)];
            d(idx_to_remove) = d(new_size);
            V.col(idx_to_remove) = V.col(new_size);
            V_dag.col(idx_to_remove) = V_dag.col(new_size);
        }

        cols.resize(static_cast<size_t>(new_size));
        d.conservativeResize(new_size);
        V.conservativeResize(Eigen::NoChange, new_size);
        V_dag.conservativeResize(Eigen::NoChange, new_size);
    }
};

} // namespace MatSubset

#endif // MAT_SUBSET_VOLUME_REMOVAL_SELECTOR_H