#ifndef MAT_SUBSET_VOLUME_REMOVAL_SELECTOR_H
#define MAT_SUBSET_VOLUME_REMOVAL_SELECTOR_H

#include <Eigen/QR> // For Eigen::HouseholderQR

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
                                               Eigen::Index k,
                                               Eigen::Index *swap_count) override {
        const Eigen::Index m = X.rows();
        const Eigen::Index n = X.cols();

        std::vector<Eigen::Index> cols(n);
        for (Eigen::Index j = 0; j < n; ++j) {
            cols[j] = j;
        }

        Eigen::HouseholderQR<Eigen::MatrixX<Scalar>> qr(X.transpose());
        Eigen::MatrixX<Scalar> V =
            (qr.householderQ() * Eigen::MatrixX<Scalar>::Identity(n, m))
                .transpose();

        Eigen::MatrixX<Scalar> V_dag = V;
        Eigen::ArrayX<Scalar> d =
            static_cast<Scalar>(1) - (V.transpose() * V_dag).diagonal().array();

        // Scratch for the loop below, allocated once at full width and used
        // through .head(active) as the active set shrinks. Written the obvious
        // way, each iteration asks the allocator for an m x n block and two
        // length-n vectors; see FrobeniusRemovalSelector for why that dominates
        // the run rather than the arithmetic.
        Eigen::VectorX<Scalar> w(m), w_dag(m), w_scaled(m);
        Eigen::VectorX<Scalar> wV_dag(n), wdV(n);

        while (cols.size() > k) {
            Eigen::Index j_max;
            Scalar d_max = d.maxCoeff(&j_max);

            // Copies, not views: removeColumn overwrites both columns below.
            w = V.col(j_max);
            w_dag = V_dag.col(j_max);

            removeColumn(cols, d, V, V_dag, j_max);

            const Eigen::Index active = static_cast<Eigen::Index>(cols.size());
            auto wV_dag_a = wV_dag.head(active);
            auto wdV_a = wdV.head(active);

            wV_dag_a.noalias() = V_dag.transpose() * w;
            d -= wV_dag_a.array().square() / d_max;

            // One gemv into wdV, then one ger into V_dag. noalias() is what
            // keeps the second in place: without it Eigen assumes V_dag may
            // appear on the right and builds the m x n product first.
            wdV_a.noalias() = V.transpose() * w_dag;
            w_scaled = w_dag / d_max;
            V_dag.noalias() += w_scaled * wdV_a.transpose();
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
    void removeColumn(std::vector<Eigen::Index> &cols, Eigen::ArrayX<Scalar> &d,
                      Eigen::MatrixX<Scalar> &V, Eigen::MatrixX<Scalar> &V_dag,
                      Eigen::Index idx_to_remove) const {

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