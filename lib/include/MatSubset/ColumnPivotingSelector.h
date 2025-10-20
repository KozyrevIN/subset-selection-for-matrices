#ifndef MAT_SUBSET_COLUNM_PIVOTING_SELECTOR_H
#define MAT_SUBSET_COLUNM_PIVOTING_SELECTOR_H

#include "SelectorBase.h" // Base class

namespace MatSubset {

/*!
 * @brief Finds a submatrix with large volume by column pivoting. Equivalent to
 * selecting `m` first pivot columns identified by QR with column pivoting.
 * @tparam Scalar The underlying scalar type (e.g., `float`, `double`).
 *
 * This selector iteratively selects a column with largest component orthogonal
 * to subspace of already selected columns. While this approach has no explicit
 * guarantee for the norm of the pseudoinverse of the selected submatrix, it
 * provides an exponential bound for its volume relative to the maximum volume
 * achievable on submatrices of such size (Çivril, Magdon-Ismail (2009) "On
 * selecting a maximum volume sub-matrix of a matrix and related problems").
 * Other selectors like `DominantSelector` and `RectMaxvolSelector` use this
 * algorithm internally to obtain the starting set of columns.
 *
 * @note This selector only supports selecting \f$ k = m \f$ columns, where \f$
 * m \f$ is the number of rows in the input matrix.
 */
template <typename Scalar>
class ColumnPivotingSelector : public SelectorBase<Scalar> {
  public:
    /*!
     * @brief Default constructor for `ColumnPivotingSelector`.
     */
    ColumnPivotingSelector() = default;

    /*!
     * @brief Gets the human-readable name of the algorithm.
     * @return The string "column pivoting".
     */
    std::string getAlgorithmName() const override { return "column pivoting"; }

  protected:
    /*!
     * @brief Core implementation for selecting a subset of \f$ k \f$ columns.
     * @param X The input matrix (dimensions \f$ m \times n \f$) from which
     * columns are to be selected. It is assumed that \f$ X \f$ is full rank
     * for theoretical guarantees.
     * @param k The number of columns to select. Must be equal to \f$ m \f$ for
     * this method.
     * @return A `std::vector` of `Eigen::Index` containing the 0-based indices
     * of the selected columns.
     */
    std::vector<Eigen::Index> selectSubsetImpl(const Eigen::MatrixX<Scalar> &X,
                                               Eigen::Index k) override {
        // This selector expects k to be equal to the number of rows
        assert(k == X.rows() && "ColumnPivotingSelector only supports k == m.");

        Eigen::MatrixX<Scalar> R = X;
        std::vector<Eigen::Index> indices;
        for (Eigen::Index i = 0; i < k; ++i) {
            Eigen::ArrayX<Scalar> gamma = R.colwise().squaredNorm();
            Eigen::Index j_max;
            Scalar gamma_max = gamma.maxCoeff(&j_max);
            indices.push_back(j_max);
            R -= R.col(j_max) * (R.col(j_max).transpose() * R) / gamma_max;
        }

        return indices;
    }
};

} // namespace MatSubset

#endif // MAT_SUBSET_COLUNM_PIVOTING_SELECTOR_H