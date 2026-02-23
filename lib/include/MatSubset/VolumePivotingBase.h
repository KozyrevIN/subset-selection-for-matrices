#ifndef MAT_SUBSET_VOLUME_PIVOTING_BASE_H
#define MAT_SUBSET_VOLUME_PIVOTING_BASE_H

#include "SelectorBase.h" // Base class

namespace MatSubset {

/*!
 * @brief Provides common functionality for subset selection by iterative column
 * pivoting to maximize volume.
 * @tparam Scalar The underlying scalar type (e.g., `float`, `double`).
 *
 * This base class implements the column pivoting algorithm which iteratively
 * selects a column with the largest component orthogonal to the subspace of
 * already selected columns. Equivalent to selecting `m` first pivot columns
 * identified by QR with column pivoting.
 *
 * While this approach has no explicit guarantee for the norm of the
 * pseudoinverse of the selected submatrix, it provides an exponential bound for
 * its volume relative to the maximum volume achievable on submatrices of such
 * size (Çivril, Magdon-Ismail (2009) "On selecting a maximum volume sub-matrix
 * of a matrix and related problems").
 *
 * Other selectors like `DominantSelector` and `RectMaxvolSelector` use this
 * algorithm internally to obtain the starting set of columns.
 *
 * @note This class is abstract and not intended for independent use. No objects
 * of this class can be created. Use derived classes like `DominantSelector` or
 * `RectMaxvolSelector` instead.
 */
template <typename Scalar>
class VolumePivotingBase : public SelectorBase<Scalar> {
  public:
    /*!
     * @brief Default constructor for `VolumePivotingBase`.
     */
    VolumePivotingBase() = default;

  protected:
    /*!
     * @brief Permutes columns of matrix \f$ X \f$ to form a well-conditioned
     * submatrix in its first m columns using volume-based column pivoting.
     * @param X The input matrix (dimensions \f$ m \times n \f$). Columns are
     * permuted in-place so that the first m columns form a highly nondegenerate
     * submatrix with large volume.
     * @return A `std::vector` of `Eigen::Index` containing the 0-based indices
     * tracking the column permutation.
     *
     * This method is intended to be called by derived classes as part of their
     * `selectSubsetImpl` implementation to obtain an initial highly
     * nondegenerate subset of m columns with large volume.
     *
     * After execution, X(:, 0:m-1) contains the selected columns, and the
     * returned indices vector tracks which original column index is now at
     * each position.
     */
    std::vector<Eigen::Index>
    selectStartingSet(Eigen::MatrixX<Scalar> &X) const {
        const Eigen::Index m = X.rows();
        const Eigen::Index n = X.cols();

        // Make a copy for orthogonalization process
        Eigen::MatrixX<Scalar> R = X;

        std::vector<Eigen::Index> indices(n);
        for (Eigen::Index j = 0; j < n; ++j) {
            indices[j] = j;
        }

        for (Eigen::Index i = 0; i < m; ++i) {
            Eigen::ArrayX<Scalar> gamma = R.colwise().squaredNorm();
            Eigen::Index j_max;
            Scalar gamma_max = gamma.tail(n - i).maxCoeff(&j_max);
            j_max += i;

            std::swap(indices[static_cast<size_t>(i)],
                      indices[static_cast<size_t>(j_max)]);
            X.col(i).swap(X.col(j_max));
            R.col(i).swap(R.col(j_max));

            R -= R.col(i) * (R.col(i).transpose() * R) / gamma_max;
        }

        return indices;
    }
};

} // namespace MatSubset

#endif // MAT_SUBSET_VOLUME_PIVOTING_BASE_H
