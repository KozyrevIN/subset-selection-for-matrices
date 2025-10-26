#ifndef MAT_SUBSET_FROBENIUS_SELECTION_SELECTOR_H
#define MAT_SUBSET_FROBENIUS_SELECTION_SELECTOR_H

#include <cmath> // For stc::copysign

#include "SelectorBase.h" // Base class

namespace MatSubset {

/*!
 * @brief Approximates subset selection problem for matrices using algorithm
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
class FrobeniusSelectionSelector : public SelectorBase<Scalar> {
  public:
    /*!
     * @brief Default constructor for `FrobeniusSelectionSelector`.
     */
    FrobeniusSelectionSelector() = default;

    /*!
     * @brief Gets the human-readable name of the algorithm.
     * @return The string "column pivoting".
     */
    std::string getAlgorithmName() const override {
        return "frobenius selection";
    }

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

        const Eigen::Index m = X.rows();
        const Eigen::Index n = X.cols();
        // This selector expects k to be equal to the number of rows
        assert(k == m && "FrobeniusSelectionSelector only supports k == m.");

        std::vector<Eigen::Index> indices(n);
        for (Eigen::Index j = 0; j < n; ++j) {
            indices[j] = j;
        }

        Eigen::MatrixX<Scalar> W = Eigen::MatrixX<Scalar>::Zero(m, n);
        Eigen::BDCSVD<Eigen::MatrixX<Scalar>> svd(X, Eigen::ComputeThinV);
        Eigen::MatrixX<Scalar> V = svd.matrixV().transpose();

        for (Eigen::Index i = 0; i < k; ++i) {
            Eigen::ArrayX<Scalar> l =
                (static_cast<Scalar>(1) +
                 W.topRows(i).colwise().squaredNorm().array()) /
                V.bottomRows(m - i).colwise().squaredNorm().array();
            Eigen::Index j_min;
            l.tail(n - i).minCoeff(&j_min);
            j_min += i;

            std::swap(indices[static_cast<size_t>(i)],
                      indices[static_cast<size_t>(j_min)]);
            V.col(i).swap(V.col(j_min));
            W.col(i).swap(W.col(j_min));

            Eigen::VectorX<Scalar> v = V.col(i).tail(m - i);
            v(0) += std::copysign(v.norm(), v(0));
            v /= v.norm();

            V.bottomRows(m - i) -= 2 * v * v.transpose() * V.bottomRows(m - i);
            W.topRows(i) -= W.col(i).head(i) * V.row(i) / V(i, i);
            W.row(i) += V.row(i) / V(i, i);
        }

        indices.resize(k);
        return indices;
    }
};

} // namespace MatSubset

#endif // MAT_SUBSET_FROBENIUS_SELECTION_SELECTOR_H