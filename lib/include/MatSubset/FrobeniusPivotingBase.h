#ifndef MAT_SUBSET_FROBENIUS_PIVOTING_BASE_H
#define MAT_SUBSET_FROBENIUS_PIVOTING_BASE_H

#include <cmath>   // For std::copysign
#include <utility> // For std::move

#include "SelectorBase.h" // Base class

namespace MatSubset {

/*!
 * @brief Provides common functionality for subset selection by providing a
 * function that finds highly nondegenerate \f$m \times m\f$ submatrix.
 * @tparam Scalar The underlying scalar type (e.g., `float`, `double`).
 *
 * This selector implements Algorithm 2 from Osinsky (2025), "Close to optimal
 * column approximation using a single SVD". It greedily selects columns that
 * minimize the Frobenius norm of the pseudoinverse of the selected submatrix.
 * Other selectors like `FrobeniusSelectionSelector` use this algorithm
 * internally to obtain the starting set of columns.
 *
 * @note This class is abstract and not intended for independent use. No objects
 * of this class can be created.
 */
template <typename Scalar>
class FrobeniusPivotingBase : public SelectorBase<Scalar> {
  public:
    /*!
     * @brief Default constructor for `FrobeniusPivotingBase`.
     */
    FrobeniusPivotingBase() = default;

  protected:
    /*!
     * @brief Permutes columns of matrix \f$ V \f$
     * to form a well-conditioned submatrix in its first m columns.
     * @param V The input matrix (dimensions \f$ m \times n \f$). This matrix is
     * expected to have orthonormal rows.
     * @param Wt_out Optional output: the algorithm's bookkeeping matrix
     * \f$ W = V_{\mathcal{S}}^{-1} V \f$, stored transposed
     * (\f$ n \times m \f$, row `j` holds \f$ W_j = V_{\mathcal{S}}^{-1} v_j
     * \f$), in the same permuted column order as the returned indices and the
     * transformed `V`. Here \f$ V_{\mathcal{S}} \f$ is the leading
     * \f$ m \times m \f$ block of the transformed `V` (upper triangular after
     * the Householder sweep). Callers can reuse it: e.g.
     * \f$ 1 + \lVert W_j \rVert^2 = 1 + v_j^{\top} (V_{\mathcal{S}}
     * V_{\mathcal{S}}^{\top})^{-1} v_j \f$.
     * @return A `std::vector` of `Eigen::Index` of permuted 0-based indices.
     *
     * This method is intended to be called by derived classes as part of their
     * `selectSubsetImpl` implementation to obtain an initial highly
     * nondegenerate subset of columns.
     */
    std::vector<Eigen::Index>
    selectStartingSet(Eigen::MatrixX<Scalar> &V,
                      Eigen::MatrixX<Scalar> *Wt_out = nullptr) const {

        const Eigen::Index m = V.rows();
        const Eigen::Index n = V.cols();

        std::vector<Eigen::Index> indices(n);
        for (Eigen::Index j = 0; j < n; ++j) {
            indices[j] = j;
        }

        // Work on the transpose: for wide V (n >> m) every per-iteration
        // sweep - the two norm scans, the Householder update, the W update -
        // touches all n candidate columns, and in the direct layout those are
        // strided row-block passes over a column-major matrix.
        Eigen::MatrixX<Scalar> Vt = V.transpose();
        Eigen::MatrixX<Scalar> Wt = Eigen::MatrixX<Scalar>::Zero(n, m);

        Eigen::ArrayX<Scalar> numer(n);
        Eigen::ArrayX<Scalar> denom(n);
        for (Eigen::Index i = 0; i < m; ++i) {
            // Pivot scores l_j = (1 + ||W_j||^2) / ||V_bottom_j||^2, both
            // norms accumulated column-by-column (contiguous n-vectors).
            numer.setOnes();
            for (Eigen::Index c = 0; c < i; ++c) {
                numer += Wt.col(c).array().square();
            }
            denom.setZero();
            for (Eigen::Index c = i; c < m; ++c) {
                denom += Vt.col(c).array().square();
            }

            Eigen::Index j_min;
            (numer.tail(n - i) / denom.tail(n - i)).minCoeff(&j_min);
            j_min += i;

            std::swap(indices[static_cast<size_t>(i)],
                      indices[static_cast<size_t>(j_min)]);
            Vt.row(i).swap(Vt.row(j_min));
            Wt.row(i).swap(Wt.row(j_min));

            Eigen::VectorX<Scalar> v = Vt.row(i).tail(m - i).transpose();
            v(0) += std::copysign(v.norm(), v(0));
            v /= v.norm();

            // Householder reflector applied from the right of Vt's trailing
            // columns (equivalently from the left of V's bottom rows): one
            // GEMV plus a rank-1 update.
            Eigen::VectorX<Scalar> y = Vt.rightCols(m - i) * v;
            Vt.rightCols(m - i).noalias() -=
                Scalar(2) * y * v.transpose();

            // W bookkeeping, transposed: columns of W are rows of Wt. The
            // pivot's W-row must be copied out first - the rank-1 update
            // rewrites it (to zero, as in the direct layout).
            const Eigen::VectorX<Scalar> alpha = Vt.col(i) / Vt(i, i);
            if (i > 0) {
                const Eigen::VectorX<Scalar> w =
                    Wt.row(i).head(i).transpose();
                Wt.leftCols(i).noalias() -= alpha * w.transpose();
            }
            Wt.col(i) = alpha;
        }

        V = Vt.transpose();
        if (Wt_out) {
            *Wt_out = std::move(Wt);
        }
        return indices;
    }
};

} // namespace MatSubset

#endif // MAT_SUBSET_FROBENIUS_PIVOTING_BASE_H