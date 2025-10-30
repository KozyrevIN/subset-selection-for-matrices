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
     * @brief Selects `m` columns with large volume using column pivoting.
     * @param X The input matrix (dimensions \f$ m \times n \f$) from which
     * columns are to be selected. It is assumed that \f$ X \f$ is full rank
     * for theoretical guarantees.
     * @return A `std::vector` of `Eigen::Index` containing the 0-based indices
     * of the selected columns.
     *
     * This method is intended to be called by derived classes as part of their
     * `selectSubsetImpl` implementation to obtain an initial highly
     * nondegenerate subset of m columns with large volume.
     */
    std::vector<Eigen::Index>
    selectStartingSet(const Eigen::MatrixX<Scalar> &X) {
        const Eigen::Index m = X.rows();

        Eigen::MatrixX<Scalar> R = X;
        std::vector<Eigen::Index> indices;
        indices.reserve(m);

        for (Eigen::Index i = 0; i < m; ++i) {
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

#endif // MAT_SUBSET_VOLUME_PIVOTING_BASE_H
