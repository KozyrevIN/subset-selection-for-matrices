#ifndef MAT_SUBSET_VOLUME_PIVOTING_BASE_H
#define MAT_SUBSET_VOLUME_PIVOTING_BASE_H

#include <cassert> // For assert

#include <Eigen/LU> // For Eigen::MatrixBase::inverse
#include <Eigen/QR> // For Eigen::HouseholderQR

#include "Enums.h"        // For MatSubset::Initialization
#include "SelectorBase.h" // Base class

namespace MatSubset {

/*!
 * @brief Provides common functionality for subset selection by iterative column
 * pivoting to maximize volume.
 * @tparam Scalar The underlying scalar type (e.g., `float`, `double`).
 *
 * This base class implements the column pivoting algorithm which iteratively
 * selects a column with the largest component orthogonal to the subspace of
 * already selected columns via CPQR. Depending on the chosen initialization
 * strategy, it then greedily selects more columns to maximize volume, and
 * optionally uses oversampling with downsampling for a better starting set.
 *
 * While this approach has no explicit guarantee for the norm of the
 * pseudoinverse of the selected submatrix, it provides an exponential bound for
 * its volume relative to the maximum volume achievable on submatrices of such
 * size (Çivril, Magdon-Ismail (2009) "On selecting a maximum volume sub-matrix
 * of a matrix and related problems").
 *
 * Other selectors like `DominantSelector`, `VolumeAddRemoveSelector` and
 * `RectMaxvolSelector` use this algorithm internally to obtain the starting set
 * of columns.
 *
 * @note This class is abstract and not intended for independent use. No objects
 * of this class can be created. Use derived classes instead.
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
     * submatrix in its first k columns using volume-based column pivoting.
     * @param X The input matrix (dimensions \f$ m \times n \f$). Columns are
     * permuted in-place so that the first k columns form a highly nondegenerate
     * submatrix with large volume.
     * @param k The number of columns to select. Must be >= m.
     * @param init The initialization strategy:
     *   - `Initialization::CPQR`: select only m columns via CPQR, then pad
     *     with the next k-m columns as-is.
     *   - `Initialization::Greedy`: select m columns via CPQR, then greedily
     *     add k-m more columns to maximize volume.
     *   - `Initialization::Advanced`: initialization involving oversampling,
     *     exchanges and downsampling. Provides lower theoretical complexity
     *     but can be suboptimal in practice.
     * @return A `std::vector` of `Eigen::Index` containing the 0-based indices
     * tracking the column permutation.
     *
     * This method is intended to be called by derived classes as part of their
     * `selectSubsetImpl` implementation to obtain an initial highly
     * nondegenerate subset of k columns with large volume.
     *
     * After execution, X(:, 0:k-1) contains the selected columns, and the
     * returned indices vector tracks which original column index is now at
     * each position.
     */
    std::vector<Eigen::Index>
    selectStartingSet(Eigen::MatrixX<Scalar> &X, Eigen::Index k,
                      Initialization init = Initialization::Greedy) const {
        const Eigen::Index m = X.rows();
        const Eigen::Index n = X.cols();

        assert(((init != Initialization::Advanced) || (n >= 2 * m - 1)) &&
               "Advanced initialization only supports n >= 2m - 1.");

        // LQ decomposition
        Eigen::HouseholderQR<Eigen::MatrixX<Scalar>> qr(X.transpose());
        X = (qr.householderQ() * Eigen::MatrixX<Scalar>::Identity(n, m))
                .transpose();

        // Pivoting for the first m columns
        std::vector<Eigen::Index> indices(n);
        for (Eigen::Index j = 0; j < n; ++j) {
            indices[j] = j;
        }

        Eigen::MatrixX<Scalar> R = X;
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

        if (init == Initialization::CPQR) {
            return indices;
        }

        // Greedy selection phase
        const Eigen::Index select_up_to =
            (init == Initialization::Advanced) ? 2 * m - 1 : k;

        Eigen::MatrixX<Scalar> Y =
            (X.leftCols(m) * X.leftCols(m).transpose()).inverse();
        Eigen::VectorX<Scalar> l = (X.transpose() * Y * X).diagonal();

        greedyAdd(X, indices, l, Y, m, select_up_to);

        if (init == Initialization::Greedy) {
            return indices;
        }

        // Exchange phase
        exchangePhase(X, indices, l, Y, k,
                      static_cast<Scalar>(4 * m - 1) / (2 * m));

        // Greedy removal/selection phase: adjust from 2m-1 columns to k
        if (k <= 2 * m - 1) {
            greedyRemove(X, indices, l, Y, 2 * m - 1, k);
        } else {
            greedyAdd(X, indices, l, Y, 2 * m - 1, k);
        }

        return indices;
    }

  private:
    /*!
     * @brief Greedily adds columns to the selected set by repeatedly picking
     * the column with the largest leverage score among the remaining ones.
     * @param X The working matrix (columns permuted in-place).
     * @param indices Column index tracker (permuted in-place).
     * @param l Leverage score vector (updated in-place).
     * @param Y The inverse Gram matrix \f$ (X_S X_S^T)^{-1} \f$ of the
     * current selection (updated in-place).
     * @param from The first position to fill (0-based).
     * @param to One past the last position to fill.
     */
    void greedyAdd(Eigen::MatrixX<Scalar> &X,
                   std::vector<Eigen::Index> &indices,
                   Eigen::VectorX<Scalar> &l, Eigen::MatrixX<Scalar> &Y,
                   Eigen::Index from, Eigen::Index to) const {

        const Eigen::Index n = X.cols();
        for (Eigen::Index t = from; t < to; ++t) {
            Eigen::Index j_max;
            Scalar l_max = l.tail(n - t).maxCoeff(&j_max);
            j_max += t;
            Eigen::VectorX<Scalar> Y_x_max = Y * X.col(j_max);

            std::swap(indices[static_cast<size_t>(t)],
                      indices[static_cast<size_t>(j_max)]);
            std::swap(l(t), l(j_max));
            X.col(t).swap(X.col(j_max));

            l -= (Y_x_max.transpose() * X).cwiseAbs2() / (1 + l_max);
            Y -= Y_x_max * Y_x_max.transpose() / (1 + l_max);
        }
    }

    /*!
     * @brief Greedily removes columns from the selected set by repeatedly
     * evicting the column with the smallest leverage score, shrinking the
     * subset from size `from` down to size `to`.
     * @param X The working matrix (columns permuted in-place).
     * @param indices Column index tracker (permuted in-place).
     * @param l Leverage score vector (updated in-place).
     * @param Y The inverse Gram matrix \f$ (X_S X_S^T)^{-1} \f$ of the
     * current selection (updated in-place).
     * @param from The current subset size to shrink from.
     * @param to The target subset size to shrink to.
     */
    void greedyRemove(Eigen::MatrixX<Scalar> &X,
                      std::vector<Eigen::Index> &indices,
                      Eigen::VectorX<Scalar> &l, Eigen::MatrixX<Scalar> &Y,
                      Eigen::Index from, Eigen::Index to) const {
        for (Eigen::Index t = from; t > to; --t) {
            Eigen::Index j_min;
            Scalar l_min = l.head(t).minCoeff(&j_min);
            Eigen::VectorX<Scalar> Y_x_min = Y * X.col(j_min);

            std::swap(indices[static_cast<size_t>(t - 1)],
                      indices[static_cast<size_t>(j_min)]);
            std::swap(l(t - 1), l(j_min));
            X.col(t - 1).swap(X.col(j_min));

            l += (Y_x_min.transpose() * X).cwiseAbs2() / (1 - l_min);
            Y += Y_x_min * Y_x_min.transpose() / (1 - l_min);
        }
    }

    /*!
     * @brief Runs the exchange phase, iteratively swapping columns in and out
     * of the k-subset to increase volume until no sufficiently improving swap
     * exists.
     * @param X The working matrix (columns permuted in-place).
     * @param indices Column index tracker (permuted in-place).
     * @param l Leverage score vector (updated in-place).
     * @param Y The inverse Gram matrix \f$ (X_S X_S^T)^{-1} \f$ of the
     * current selection (updated in-place).
     * @param k The size of the current subset (columns 0..k-1 are selected).
     * @param vol_bound The volume improvement threshold below which the loop
     * terminates.
     */
    void exchangePhase(Eigen::MatrixX<Scalar> &X,
                       std::vector<Eigen::Index> &indices,
                       Eigen::VectorX<Scalar> &l, Eigen::MatrixX<Scalar> &Y,
                       Eigen::Index k, Scalar vol_bound) const {
        const Eigen::Index n = X.cols();
        while (true) {
            // Find the best candidate column to add (from outside the k-subset)
            Eigen::Index s;
            Scalar l_s = l.tail(n - k).maxCoeff(&s);
            s += k;

            // Find the weakest column to remove (from within the k-subset)
            Eigen::Index r;
            Scalar l_r = l.head(k).minCoeff(&r);

            if ((1 + l_s) * (1 - l_r) <= vol_bound) {
                break;
            }

            // Apply rank-1 update for adding column s, then downdate for
            // removing column r
            Eigen::VectorX<Scalar> Y_x_s = Y * X.col(s);
            Eigen::VectorX<Scalar> l_prime =
                l - (Y_x_s.transpose() * X).cwiseAbs2() / (1 + l_s);
            Eigen::MatrixX<Scalar> Y_prime =
                Y - Y_x_s * Y_x_s.transpose() / (1 + l_s);

            Eigen::VectorX<Scalar> Y_prime_x_r = Y_prime * X.col(r);
            l = l_prime +
                (Y_prime_x_r.transpose() * X).cwiseAbs2() / (1 - l_r);
            Y = Y_prime + Y_prime_x_r * Y_prime_x_r.transpose() / (1 - l_r);

            std::swap(indices[static_cast<size_t>(r)],
                      indices[static_cast<size_t>(s)]);
            std::swap(l(r), l(s));
            X.col(r).swap(X.col(s));
        }
    }
};

} // namespace MatSubset

#endif // MAT_SUBSET_VOLUME_PIVOTING_BASE_H
