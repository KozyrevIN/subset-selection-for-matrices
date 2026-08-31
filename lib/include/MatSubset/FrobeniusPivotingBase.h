#ifndef MAT_SUBSET_FROBENIUS_PIVOTING_BASE_H
#define MAT_SUBSET_FROBENIUS_PIVOTING_BASE_H

#include <cmath>   // For std::copysign
#include <limits>  // For std::numeric_limits
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
     * @brief Permutes columns of matrix \f$ V \f$ to form a well-conditioned
     * submatrix in its first \f$ m \f$ columns.
     * @param V The input matrix (dimensions \f$ m \times n \f$), expected to
     * have orthonormal rows. Overwritten with the permuted matrix, reflected
     * so that its leading \f$ m \times m \f$ block is upper triangular.
     * @param M_out Optional output: the inverse Gram
     * \f$ M = (V_{\mathcal{S}} V_{\mathcal{S}}^{\top})^{-1} \f$ of the
     * selected block (\f$ m \times m \f$, symmetric).
     * @param d_out Optional output: the scores
     * \f$ d_j = 1 + v_j^{\top} M v_j \f$ for every column, in the same
     * permuted order as the returned indices.
     * @return A `std::vector` of `Eigen::Index` of permuted 0-based indices.
     *
     * This method is intended to be called by derived classes as part of their
     * `selectSubsetImpl` implementation to obtain an initial highly
     * nondegenerate subset of columns. Step \f$ i \f$ takes
     * \f$ \arg\min_j d_j / \lVert V_{i:,j} \rVert^2 \f$, and neither
     * sweep is ever rebuilt from scratch:
     *
     * - \f$ M \f$ **gains a row and a column and keeps its existing block**.
     *   The selected block is triangular, so extending it leaves the inverse
     *   of its leading part alone. With \f$ u = V_{:i,\text{piv}} \f$,
     *   \f$ \tau = V_{i,\text{piv}} \f$, \f$ z = M u \f$ and
     *   \f$ s = d_{\text{piv}} = 1 + u^{\top} M u \f$ (already carried),
     *   \f$ M \gets \begin{pmatrix} M & -z/\tau \\ -z^{\top}/\tau &
     *   s/\tau^2\end{pmatrix} \f$ - an \f$ O(i^2) \f$ fill, not a
     *   factorisation.
     * - \f$ d \f$ follows from one candidate sweep \f$ g_j = z^{\top} v_j \f$:
     *   \f$ d_j \gets d_j - 2\alpha_j g_j + \alpha_j^2 s \f$ with
     *   \f$ \alpha_j = V_{i,j}/\tau \f$.
     * - \f$ \lVert V_{i:,j} \rVert^2 \f$ survives the reflector untouched (it
     *   is orthogonal on exactly the slice it acts on), so advancing \f$ i \f$
     *   only drops the coordinate that just left the slice.
     *
     * Carrying \f$ W = V_{\mathcal{S}}^{-1} V \f$ instead, and rescanning
     * both norm sweeps every step, computes the same pivots for
     * \f$ \approx 2.5\,nm^2 \f$ work against \f$ \approx 1.5\,nm^2 \f$
     * here, and needs an extra \f$ n \times m \f$ array to do it.
     */
    std::vector<Eigen::Index>
    selectStartingSet(Eigen::MatrixX<Scalar> &V,
                      Eigen::MatrixX<Scalar> *M_out = nullptr,
                      Eigen::ArrayX<Scalar> *d_out = nullptr) const {

        const Eigen::Index m = V.rows();
        const Eigen::Index n = V.cols();

        std::vector<Eigen::Index> indices(n);
        for (Eigen::Index j = 0; j < n; ++j) {
            indices[j] = j;
        }

        // Work on the transpose: for wide V (n >> m) every per-iteration
        // sweep touches all n candidates, and in the direct layout those
        // are strided row-block passes over a column-major matrix.
        Eigen::MatrixX<Scalar> Vt = V.transpose();
        Eigen::MatrixX<Scalar> M = Eigen::MatrixX<Scalar>::Zero(m, m);

        Eigen::ArrayX<Scalar> d = Eigen::ArrayX<Scalar>::Ones(n);
        Eigen::ArrayX<Scalar> denom = Vt.rowwise().squaredNorm().array();

        const Scalar tiny = std::numeric_limits<Scalar>::min();

        Eigen::VectorX<Scalar> h(m);
        Eigen::VectorX<Scalar> y(n);
        Eigen::VectorX<Scalar> z(m);
        Eigen::VectorX<Scalar> g(n);
        Eigen::ArrayX<Scalar> alpha(n);
        for (Eigen::Index i = 0; i < m; ++i) {
            // A candidate whose remaining norm has underflowed is one the
            // selected columns already explain, so letting its score blow up
            // is the right answer; the clamp only keeps the division finite.
            Eigen::Index j_min;
            (d.tail(n - i) / denom.tail(n - i).max(tiny)).minCoeff(&j_min);
            j_min += i;

            if (j_min != i) {
                std::swap(indices[static_cast<size_t>(i)],
                          indices[static_cast<size_t>(j_min)]);
                Vt.row(i).swap(Vt.row(j_min));
                std::swap(d(i), d(j_min));
                std::swap(denom(i), denom(j_min));
            }

            // The sign that grows the leading entry: the pivot rule looks
            // for large ||V(i:, j)||, so x is often already close to e_1 and
            // the cancelling choice would lose accuracy exactly there.
            h.head(m - i) = Vt.row(i).tail(m - i).transpose();
            const Scalar nx = h.head(m - i).norm();
            h(0) += std::copysign(nx, h(0));
            h.head(m - i) /= h.head(m - i).norm();
            y.noalias() = Vt.rightCols(m - i) * h.head(m - i);
            Vt.rightCols(m - i).noalias() -=
                Scalar(2) * y * h.head(m - i).transpose();

            // ||V(i:, j)||^2 -> ||V(i+1:, j)||^2
            denom -= Vt.col(i).array().square();

            const Scalar tau = Vt(i, i); // = -sign(x_0) nx, so |tau| = nx
            const Scalar s = d(i);       // = 1 + u^T M u, already carried
            alpha = Vt.col(i).array() / tau;
            if (i > 0) {
                z.head(i).noalias() =
                    M.topLeftCorner(i, i) * Vt.row(i).head(i).transpose();
                g.noalias() = Vt.leftCols(i) * z.head(i);
                // The clamp absorbs cancellation only: d >= 1 by construction.
                d = (d - Scalar(2) * alpha * g.array() + alpha.square() * s)
                        .max(Scalar(1));
                M.col(i).head(i) = -z.head(i) / tau;
                M.row(i).head(i) = M.col(i).head(i).transpose();
            } else {
                d = (d + alpha.square() * s).max(Scalar(1));
            }
            M(i, i) = s / (nx * nx);
        }

        V = Vt.transpose();
        if (M_out) {
            *M_out = std::move(M);
        }
        if (d_out) {
            *d_out = std::move(d);
        }
        return indices;
    }
};

} // namespace MatSubset

#endif // MAT_SUBSET_FROBENIUS_PIVOTING_BASE_H