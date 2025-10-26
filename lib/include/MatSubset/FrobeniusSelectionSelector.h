#ifndef MAT_SUBSET_FROBENIUS_SELECTION_SELECTOR_H
#define MAT_SUBSET_FROBENIUS_SELECTION_SELECTOR_H

#include "SelectorBase.h" // Base class

namespace MatSubset {

/*!
 * @brief Approximates subset selection problem for matrices iteratively
 * selecting columns that maximize the Frobenius norm of the pseudoinverse of
 * the selected submatrix.
 * @tparam Scalar The underlying scalar type (e.g., `float`, `double`).
 *
 * Implement a novel greedy selection algorithm
 *
 * @note This class is abstract and not intended for independent use. No objects
 * of this class can be created.
 */
template <typename Scalar>
class FrobeniusPivotingSelector : public SelectorBase<Scalar> {
  public:
    /*!
     * @brief Default constructor for `FrobeniusPivotingSelector`.
     */
    FrobeniusPivotingSelector() = default;

    /*!
     * @brief Gets the human-readable name of the algorithm.
     * @return The string "frobenius selection".
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
        
    }
};

} // namespace MatSubset

#endif // MAT_SUBSET_FROBENIUS_SELECTION_SELECTOR_H