#ifndef MAT_SUBSET_VOLUME_PIVOTING_BASE_H
#define MAT_SUBSET_VOLUME_PIVOTING_BASE_H

#include <cassert> // For assert

#include <Eigen/LU> // For Eigen::MatrixBase::inverse
#include <Eigen/QR> // For Eigen::HouseholderQR

#include "SelectorBase.h" // Base class

namespace MatSubset {

/*!
 * @brief Provides common functionality for subset selection by iterative column
 * pivoting to maximize volume.
 * @tparam Scalar The underlying scalar type (e.g., `float`, `double`).
 *
 * This base class implements the column pivoting algorithm which iteratively
 * selects a column with the largest component orthogonal to the subspace of
 * already selected columns via CPQR. Next it greedily selects k + oversampling
 * columns to maximize volume. After that it drops oversampling columns again
 * greedily maximizing the volume of the submatrix in the process.
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
     * submatrix in its first m columns using volume-based column pivoting.
     * @param X The input matrix (dimensions \f$ m \times n \f$). Columns are
     * permuted in-place so that the first k columns form a highly nondegenerate
     * submatrix with large volume.
     * @param k The number of columns to select. Must be >= m. Defaults to m.
     * @param oversampling The number of extra columns to greedily add before
     * removing back down to k. Must be >= 0. Defaults to 0.
     * @return A `std::vector` of `Eigen::Index` containing the 0-based indices
     * tracking the column permutation.
     *
     * This method is intended to be called by derived classes as part of their
     * `selectSubsetImpl` implementation to obtain an initial highly
     * nondegenerate subset of m columns with large volume.
     *
     * After execution, X(:, 0:k - 1) contains the selected columns, and the
     * returned indices vector tracks which original column index is now at
     * each position.
     */
    std::vector<Eigen::Index>
    selectStartingSet(Eigen::MatrixX<Scalar> &X, Eigen::Index k = -1,
                      Eigen::Index oversampling = 0) const {
        const Eigen::Index m = X.rows();
        const Eigen::Index n = X.cols();

        if (k < 0) {
            k = m;
        }

        assert(k >= m && "k must be >= m.");
        assert(oversampling >= 0 && "Oversampling must be >= 0");

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

        // Selection for columns m + 1...k
        if (k + oversampling > m) {
            // Initialize l and Y
            Eigen::MatrixX<Scalar> Y =
                (X.leftCols(m) * X.leftCols(m).transpose()).inverse();
            Eigen::VectorX<Scalar> l = (X.transpose() * Y * X).diagonal();

            // Greedy selection phase
            for (Eigen::Index t = m; t < k + oversampling; ++t) {
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

            // Greedy removal phase
            for (Eigen::Index t = k + oversampling; t > k; --t) {
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

        return indices;
    }
};

} // namespace MatSubset

#endif // MAT_SUBSET_VOLUME_PIVOTING_BASE_H
