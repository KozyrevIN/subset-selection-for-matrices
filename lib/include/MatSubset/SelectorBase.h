#ifndef MAT_SUBSET_SELECTOR_BASE_H
#define MAT_SUBSET_SELECTOR_BASE_H

#include <cassert> // For assert
#include <string>  // For std::string
#include <vector>  // For std::vector

#include <Eigen/Core> // For vectors and matrices

#include "Enums.h" // For MatSubset::Norm

namespace MatSubset {

/*!
 * @brief Base class for algorithms that approximate the subset selection
 * problem for matrices.
 * @tparam Scalar The underlying scalar type of the matrix elements (e.g.,
 * float, double).
 *
 * This base class defines the common interface for various algorithms that
 * perform subset selection for matrices. Derived classes are expected to
 * implement specific strategies for selecting \f$ k \f$ columns from an \f$ m
 * \times n \f$ input matrix~\f$ X \f$.
 *
 * The problem typically assumes the input matrix \f$ X \f$ is of full rank
 * (i.e., rank \f$ m \f$). While algorithms might function on rank-deficient
 * matrices, theoretical guarantees or optimal behavior might not hold. Checking
 * for full rank at runtime can be computationally expensive and is generally
 * not performed by these selectors.
 *
 * The selection logic is implemented by overriding the `selectSubsetImpl`
 * method. The bound calculation logic is implemented by overriding `boundImpl`.
 * Common precondition checks are handled in the public-facing methods,
 * enforcing \f$ m, k, n \ge 1 \f$ and \f$ m \le k
 * \le n \f$.
 */
template <typename Scalar> class SelectorBase {
  public:
    /*! @brief Default constructor. */
    SelectorBase() = default;

    /*! @brief Virtual destructor to ensure proper cleanup of derived classes.
     */
    virtual ~SelectorBase() = default;

    /*!
     * @brief Gets the human-readable name of the subset selection algorithm.
     * @return A string representing the algorithm's name.
     * Derived classes should override this to return their specific algorithm
     * name. The default implementation returns "first k columns".
     */
    virtual std::string getAlgorithmName() const { return "first k columns"; }

    /*!
     * @brief Selects a subset of \f$ k \f$ columns from the input matrix \f$ X
     * \f$. (Public Interface)
     * @param X The input matrix (dimensions \f$ m \times n \f$) from which
     * columns are to be selected. It is assumed that \f$ X \f$ is full rank
     * for theoretical guarantees.
     * @param k The number of columns to select.
     * @return A `std::vector` of `Eigen::Index` containing the 0-based indices
     * of the selected columns.
     *
     * This method performs common precondition checks and then calls
     * `selectSubsetImpl` for the actual selection logic. Preconditions: \f$
     * m, k, n \ge 1 \f$ and \f$ m \le k \le n \f$.
     */
    [[nodiscard]] std::vector<Eigen::Index>
    selectSubset(const Eigen::MatrixX<Scalar> &X, Eigen::Index k) {
        // Common precondition checks
        assert(
            X.rows() >= 1 && X.cols() >= 1 && k >= 1 &&
            "Matrix dimensions (m, n) and k must be strictly positive (>= 1).");
        assert(X.rows() <= k && k <= X.cols() &&
               "Subset selection constraint violated: m <= k <= n must hold.");

        return selectSubsetImpl(X, k); // Call the virtual implementation
    }

    /*!
     * @brief Calculates a theoretical lower bound related to the selected
     * submatrix. (Public Interface)
     * @tparam norm The type of matrix norm (`Norm::Frobenius` or
     * `Norm::Spectral`).
     * @param X The input matrix (dimensions \f$ m \times n \f$).
     * @param k The number of columns that would be selected.
     * @return A Scalar value representing the calculated bound.
     *
     * The bound is a lower limit on a ratio
     * \f$ \lVert X^{\dag} \rVert^{2}/\lVert X_{\mathcal{S}}^{\dag}
     * \rVert^{2} \f$, where \f$ X^{\dag} \f$ is the Moore-Penrose pseudoinverse
     * of \f$ X \f$ and \f$ X_{\mathcal{S}}^{\dag} \f$ is the pseudoinverse of
     * the submatrix formed by selected columns \f$ \mathcal{S} \f$.
     * Preconditions: \f$ m, k, n \ge 1 \f$ and \f$ m \le k \le n \f$.
     */
    template <Norm norm>
    [[nodiscard]] Scalar bound(const Eigen::MatrixX<Scalar> &X,
                               Eigen::Index k) const {

        return bound<norm>(X.rows(), X.cols(), k);
    }

    /*!
     * @brief Calculates a theoretical lower bound based on matrix dimensions.
     * (Public Interface)
     * @tparam norm The type of matrix norm (`Norm::Frobenius` or
     * `Norm::Spectral`).
     * @param m The number of rows in the matrix.
     * @param n The number of columns in the matrix.
     * @param k The number of columns that would be selected.
     * @return A Scalar value representing the calculated bound.
     *
     * The bound is a lower limit on a ratio
     * \f$ \lVert X^{\dag} \rVert^{2}/\lVert X_{\mathcal{S}}^{\dag}
     * \rVert^{2} \f$, where \f$ X^{\dag} \f$ is the Moore-Penrose pseudoinverse
     * of \f$ X \f$ and \f$ X_{\mathcal{S}}^{\dag} \f$ is the pseudoinverse of
     * the submatrix formed by selected columns \f$ \mathcal{S} \f$.
     * Preconditions: \f$ m, k, n \ge 1 \f$ and \f$ m \le k \le n \f$.
     */
    template <Norm norm>
    [[nodiscard]] Scalar bound(Eigen::Index m, Eigen::Index n,
                               Eigen::Index k) const {
        static_assert(norm == Norm::Frobenius || norm == Norm::Spectral,
                      "SelectorBase::bound: This norm is unsupported for the "
                      "bound calculation!");

        assert(m >= 1 && n >= 1 && k >= 1 &&
               "Matrix dimensions (m, n) and k must be strictly positive (>= "
               "1) for bound calculation.");
        assert(m <= k && k <= n &&
               "Subset selection constraint violated: m <= k <= n must hold "
               "for bound calculation.");

        return boundImpl(m, n, k, norm);
    }

  protected:
    /*!
     * @brief Core implementation for selecting a subset of \f$ k \f$ columns.
     * @param X The \f$ m \times n \f$ input matrix \f$ X \f$. Assumed to be
     * validated by public `selectSubset`.
     * @param k The number of columns to select. Assumed to be validated by
     * public `selectSubset`.
     * @return A std::vector of `Eigen::Index` of selected column indices.
     *
     * Derived classes MUST override this method to implement their specific
     * selection logic. The default base class implementation selects the first
     * \f$ k \f$ columns.
     */
    virtual std::vector<Eigen::Index>
    selectSubsetImpl(const Eigen::MatrixX<Scalar> &X, Eigen::Index k) {
        // No assertions here as they are handled by the public selectSubset
        // k >= 1 is guaranteed by assertions in public selectSubset.
        std::vector<Eigen::Index> cols(static_cast<size_t>(k));
        for (Eigen::Index i = 0; i < k; ++i) {
            cols[static_cast<size_t>(i)] = i;
        }
        return cols;
    }

    /*!
     * @brief Core implementation for calculating the bound.
     * @param m Number of rows. Assumed to be validated by public `bound`
     * methods.
     * @param n Number of columns. Assumed to be validated by public `bound`
     * methods.
     * @param k Number of selected columns. Assumed to be validated by public
     * `bound` methods.
     * @param norm The norm type. Assumed to be validated by public `bound`
     * methods.
     * @return The calculated bound value.
     *
     * Derived classes should override this to provide specific bound
     * calculations. The default implementation returns \f$ 0 \f$.
     */
    virtual Scalar boundImpl([[maybe_unused]] Eigen::Index m,
                             [[maybe_unused]] Eigen::Index n,
                             [[maybe_unused]] Eigen::Index k,
                             [[maybe_unused]] Norm norm) const {
        // Parameters are marked `[[maybe_unused]]` because this default
        // implementation doesn't use them. Derived classes will. No assertions
        // here as they are handled by the public bound methods.
        return static_cast<Scalar>(0);
    }
};

} // namespace MatSubset

#endif // MAT_SUBSET_SELECTOR_BASE_H