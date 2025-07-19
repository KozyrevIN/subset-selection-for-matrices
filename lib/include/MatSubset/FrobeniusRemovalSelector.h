#ifndef MAT_SUBSET_FROBENIUS_REMOVAL_SELECTOR_H
#define MAT_SUBSET_FROBENIUS_REMOVAL_SELECTOR_H

#include <Eigen/QR>  // For completeOrthogonalDecomposition
#include <Eigen/SVD> // For Eigen::BDCSVD

#include "SelectorBase.h" // Base class

namespace MatSubset {

/*!
 * @brief Approximates subset selection problem for matrices using
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
 * The `eps` parameter is used as a threshold for values in
 * denominator, to ensure numerical stability.
 */
template <typename Scalar>
class FrobeniusRemovalSelector : public SelectorBase<Scalar> {
  public:
    /*!
     * @brief Constructor for `FrobeniusRemovalSelector`.
     * @param eps Small positive tolerance value. Defaults to `1e-6`.
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
        Eigen::VectorX<Scalar> S_inv2 =
            svd.singularValues().array().inverse().square();

        Eigen::MatrixX<Scalar> V_dag =
            V.completeOrthogonalDecomposition().pseudoInverse().transpose();
        Eigen::ArrayX<Scalar> l =
            (V_dag.transpose() * S_inv2.asDiagonal() * V_dag).diagonal();
        Eigen::ArrayX<Scalar> d = (V.transpose() * V_dag).diagonal().array();

        while (cols.size() > k) {

            Eigen::Index j_min = 0;
            while (j_min < cols.size() &&
                   d(j_min) >= static_cast<Scalar>(1) - eps_) {
                ++j_min;
            }

            assert(j_min < cols.size() &&
                   "Have not found a column with d_j < 1 - eps.");

            for (Eigen::Index j = j_min + 1; j < cols.size(); ++j) {
                if (d(j) < static_cast<Scalar>(1) - eps_ &&
                    l(j) + l(j_min) * d(j) < l(j_min) + l(j) * d(j_min)) {
                    j_min = j;
                }
            }

            Eigen::VectorX<Scalar> w = V.col(j_min);
            Eigen::VectorX<Scalar> w_dag = V_dag.col(j_min);
            Scalar denom = static_cast<Scalar>(1) - d(j_min);

            removeColumn(cols, l, d, V, V_dag, j_min);

            Eigen::ArrayX<Scalar> mul_1 = w.transpose() * V_dag;
            Eigen::ArrayX<Scalar> mul_2 =
                w_dag.transpose() * S_inv2.asDiagonal() * V_dag;
            Scalar mul_3 =
                (w_dag.transpose() * S_inv2.asDiagonal() * w_dag).value();

            d += mul_1.square() / denom;
            mul_1 /= denom;
            l += mul_1 * (2 * mul_2 + mul_1 * mul_3);

            V_dag += w_dag * (w_dag.transpose() * V) / denom;
        }

        return cols;
    }

    /*!
     * @brief Calculates the theoretical bound for Frobenius removal algorithm.
     * @param m The number of rows in the matrix.
     * @param n The number of columns in the matrix.
     * @param k The number of columns that would be selected.
     * @param norm The type of matrix norm (`Norm::Frobenius` or
     * `Norm::Spectral`).
     * @return A `Scalar` value representing the calculated lower bound on the
     * ratio \f$ \lVert X^{\dag} \rVert^{2}/\lVert X_{\mathcal{S}}^{\dag}
     * \rVert^{2} \f$.
     *
     * The bound is calculated based on the theorem 3.1 in Avron and Boutsidis
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
     * @param cols Vector of original column indices.
     * @param l Array of l-scores (numerators for the removal criterion).
     * @param d Array of d-scores (denominators for the removal
     * criterion).
     * @param V Matrix of active \f$ V \f$ columns (from SVD).
     * @param V_dag Matrix of active \f$ V^{\dag} \f$ (pseudoinverse related)
     * columns.
     * @param idx_to_remove The 0-based index *within the current active set* to
     * remove.
     */
    void removeColumn(std::vector<Eigen::Index> &cols, Eigen::ArrayX<Scalar> &l,
                      Eigen::ArrayX<Scalar> &d, Eigen::MatrixX<Scalar> &V,
                      Eigen::MatrixX<Scalar> &V_dag,
                      Eigen::Index idx_to_remove) {

        // `cols` is the vector of original indices, its size is the
        // current active_size.
        Eigen::Index new_size = static_cast<Eigen::Index>(cols.size()) - 1;
        // new_size will be >= k (target selection size) >= m (rows) >= 1.

        if (idx_to_remove < new_size) {
            cols[static_cast<size_t>(idx_to_remove)] =
                cols[static_cast<size_t>(new_size)];
            l(idx_to_remove) = l(new_size);
            d(idx_to_remove) = d(new_size);
            V.col(idx_to_remove) = V.col(new_size);
            V_dag.col(idx_to_remove) = V_dag.col(new_size);
        }

        cols.resize(static_cast<size_t>(new_size));
        l.conservativeResize(new_size);
        d.conservativeResize(new_size);
        V.conservativeResize(Eigen::NoChange, new_size);
        V_dag.conservativeResize(Eigen::NoChange, new_size);
    }
};

} // namespace MatSubset

#endif // MAT_SUBSET_FROBENIUS_REMOVAL_SELECTOR_H