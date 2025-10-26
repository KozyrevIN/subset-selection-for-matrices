#ifndef MAT_SUBSET_FROBENIUS_PIVOTING_BASE_H
#define MAT_SUBSET_FROBENIUS_PIVOTING_BASE_H

#include <cmath> // For std::copysign

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

    /*!
     * @brief Gets the human-readable name of the algorithm.
     * @return The string representing the specific algorithm name.
     *
     * Derived classes must override this method to provide their specific
     * algorithm name.
     */
    virtual std::string getAlgorithmName() const override = 0;

  protected:
    /*!
     * @brief Selects a starting set of \f$ m \f$ columns that form a
     * well-conditioned submatrix.
     * @param X The input matrix (dimensions \f$ m \times n \f$) from which
     * columns are to be selected.
     * @return A `std::vector` of `Eigen::Index` containing the 0-based indices
     * of \f$ m \f$ selected columns.
     *
     * This method is intended to be called by derived classes as part of their
     * `selectSubsetImpl` implementation to obtain an initial highly
     * nondegenerate subset of columns.
     *
     * @note The input matrix must have orthonormal rows for algorithm to work
     * as intended.
     */
    std::vector<Eigen::Index>
    selectStartingSet(const Eigen::MatrixX<Scalar> &X) {

        const Eigen::Index m = X.rows();
        const Eigen::Index n = X.cols();

        std::vector<Eigen::Index> indices(n);
        for (Eigen::Index j = 0; j < n; ++j) {
            indices[j] = j;
        }

        Eigen::MatrixX<Scalar> W = Eigen::MatrixX<Scalar>::Zero(m, n);
        Eigen::BDCSVD<Eigen::MatrixX<Scalar>> svd(X, Eigen::ComputeThinV);
        Eigen::MatrixX<Scalar> V = svd.matrixV().transpose();

        for (Eigen::Index i = 0; i < m; ++i) {
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

        indices.resize(m);
        return indices;
    }
};

} // namespace MatSubset

#endif // MAT_SUBSET_FROBENIUS_PIVOTING_BASE_H