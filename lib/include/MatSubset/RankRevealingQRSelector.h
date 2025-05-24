#ifndef MAT_SUBSET_RANK_REVEALING_QR_SELECTOR_H
#define MAT_SUBSET_RANK_REVEALING_QR_SELECTOR_H

#include <Eigen/QR> // For Eigen::ColPivHouseholderQR

#include "SelectorBase.h" // Base class

namespace MatSubset {

/*!
 * @brief Class for approximating subset selection problem for matrices using
 * Rank-Revealing QR (RRQR) factorization with column pivoting.
 * @tparam Scalar The underlying scalar type (e.g., `float`, `double`).
 *
 * This selector utilizes Eigen's column-pivoted Householder QR decomposition
 * (\f$ AP = QR \f$) to identify a set of \f$ k \f$ columns. The permutation \f$
 * P \f$ determined by the pivoting strategy indicates which columns of the
 * original matrix \f$ A \f$ (here, \f$ X \f$) are chosen to form a
 * well-conditioned basis.
 *
 * @note This selector **only supports selecting \f$ k = m \f$
 * columns**, where \f$ m \f$ is the number of rows (and thus the rank, assuming
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
     * @brief Core implementation for selecting a subset of \f$ k \f$ columns
     * using RRQR.
     * @param X The \f$ m \times n \f$ input matrix \f$ X \f$.
     * @param k The number of columns to select. **Must be equal to \f$ X.rows()
     * \f$ for this selector.**
     * @return A `std::vector` of `Eigen::Index` of selected column indices.
     */
    std::vector<Eigen::Index> selectSubsetImpl(const Eigen::MatrixX<Scalar> &X,
                                               Eigen::Index k) override {
        // This selector expects k to be equal to the number of rows (the rank
        // for a full row rank matrix)
        assert(k == X.rows() &&
               "RankRevealingQRSelector only supports k == X.rows().");

        Eigen::ColPivHouseholderQR<Eigen::MatrixX<Scalar>> qr(X);

        // qr.permutationMatrix().indices()(j) gives the original index of the
        // column that was permuted into the j-th position.
        // We select the first k (== X.rows()) such columns.
        auto perm_indices = qr.permutationMatrix().indices();

        std::vector<Eigen::Index> selected_indices(k);
        for (Eigen::Index j = 0; j < k; ++j) {
            selected_indices[static_cast<size_t>(j)] = perm_indices(j);
        }

        return selected_indices;
    }
};

} // namespace MatSubset

#endif // MAT_SUBSET_RANK_REVEALING_QR_SELECTOR_H