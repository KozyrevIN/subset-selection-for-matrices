#ifndef MAT_SUBSET_RANK_REVEALING_QR_SELECTOR_H
#define MAT_SUBSET_RANK_REVEALING_QR_SELECTOR_H

#include <Eigen/QR> // For Eigen::ColPivHouseholderQR

#include "SelectorBase.h" // Base class

namespace MatSubset {

/*!
 * @brief Approximates subset selection problem for matrices using
 * Rank-Revealing QR (RRQR) factorization with column pivoting.
 * @tparam Scalar The underlying scalar type (e.g., `float`, `double`).
 *
 * This selector utilizes Eigen's column-pivoted Householder QR decomposition
 * (\f$ AP = QR \f$) to identify a set of \f$ k \f$ columns. The permutation \f$
 * P \f$ determined by the pivoting strategy indicates which columns of the
 * original matrix \f$ A \f$ (here, \f$ X \f$) are chosen to form a
 * well-conditioned basis.
 *
 * @note This selector only supports selecting \f$ k = m \f$
 * columns, where \f$ m \f$ is the number of rows (and thus the rank, assuming
 * \f$ X \f$ is full row rank). The first \f$ k \f$ columns chosen by the
 * pivoting are returned.
 */
template <typename Scalar>
class RankRevealingQRSelector : public SelectorBase<Scalar> {
  public:
    /*!
     * @brief Default constructor for `RankRevealingQRSelector`.
     */
    RankRevealingQRSelector() = default;

    /*!
     * @brief Gets the human-readable name of the algorithm.
     * @return The string "rank-revealing QR".
     */
    std::string getAlgorithmName() const override {
        return "rank-revealing QR";
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
        // This selector expects k to be equal to the number of rows (the rank
        // for a full row rank matrix)
        assert(k == X.rows() &&
               "RankRevealingQRSelector only supports k == m.");

        Eigen::ColPivHouseholderQR<Eigen::MatrixX<Scalar>> qr(X);
        Eigen::MatrixX<Scalar> P = qr.colsPermutation();

        std::vector<Eigen::Index> indices(k);

        for (int j = 0; j < k; ++j) {
            int i = 0;
            for (; std::abs(P(i, j)) == 0; ++i)
                indices[j] = i;
        }

        std::sort(indices.begin(), indices.end());

        return indices;
    }
};

} // namespace MatSubset

#endif // MAT_SUBSET_RANK_REVEALING_QR_SELECTOR_H