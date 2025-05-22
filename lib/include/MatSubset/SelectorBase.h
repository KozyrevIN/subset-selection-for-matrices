#ifndef MAT_SUBSET_SELECTOR_BASE_H 
#define MAT_SUBSET_SELECTOR_BASE_H

#include <string>
#include <vector>

#include <Eigen/Core>

#include "Enums.h" 

namespace MatSubset {

/*!
 * @brief Base class for algorithms that select a subset of columns from a matrix.
 * @tparam scalar The underlying scalar type of the matrix elements (e.g., float, double).
 *
 * This abstract class defines the common interface for various column selection
 * algorithms. Derived classes are expected to implement specific strategies
 * for selecting `k` columns from an `m x n` matrix `X`, where `n >= k >= m`.
 *
 * The default implementation of `selectSubset` provides a placeholder
 * behavior (selecting the first `k` columns) and should be overridden by
 * concrete selector implementations.
 *
 * The `bound` methods provide a way to estimate a theoretical lower bound
 * related to the quality of the selected submatrix, typically involving
 * the ratio of pseudoinverse norms. The specific interpretation of this bound
 * depends on the context of the column selection problem (e.g., low-rank approximation,
 * numerical stability).
 */
template <typename scalar> // Changed 'scalar' to 'scalar' for common C++ template naming
class SelectorBase {
  public:
    /*!
     * @brief Default constructor.
     */
    SelectorBase() = default; // Using = default for trivial constructor

    /*!
     * @brief Virtual destructor to ensure proper cleanup of derived classes.
     */
    virtual ~SelectorBase() = default; // Important for polymorphic base classes

    /*!
     * @brief Gets the human-readable name of the column selection algorithm.
     * @return A string representing the algorithm's name.
     *
     * Derived classes should override this to return their specific algorithm name.
     * The default implementation returns "random columns", which might be a placeholder
     * or indicate a basic default strategy if not overridden.
     * @note The current default implementation name "random columns" does not match
     *       the default `selectSubset` behavior (first k columns). Consider aligning these.
     */
    virtual std::string getAlgorithmName() const { return "random columns"; } 

    /*!
     * @brief Selects a subset of `k` columns from the input matrix `X`.
     * @param X The `m x n` input matrix from which columns are to be selected.
     *          It is expected that `X.rows() <= X.cols()`.
     * @param k The number of columns to select. Must satisfy `X.rows() <= k <= X.cols()`.
     * @return A std::vector of `Eigen::Index` containing the 0-based indices of the
     *         `k` selected columns. The order of indices in the returned vector is
     *         not guaranteed unless specified by a derived class.
     * @throw std::invalid_argument If `k` is not within the valid range
     *        (e.g., `k < X.rows()` or `k > X.cols()`), or if `X` has invalid dimensions.
     *        (Note: Current implementation does not throw but derived classes should consider this.)
     *
     * This is a virtual method intended to be overridden by derived classes
     * implementing specific selection strategies. The default base class
     * implementation selects the first `k` columns (indices 0 to `k-1`).
     */
    virtual std::vector<Eigen::Index>
    selectSubset(const Eigen::MatrixX<scalar>& X, Eigen::Index k) {
        // Basic input validation (derived classes might add more specific checks)
        if (k < 0 || k > X.cols() || (X.rows() > 0 && k < X.rows())) { // Assuming k >= m is a common requirement
             throw std::invalid_argument("Invalid number of columns k requested. "
                                       "Ensure X.rows() <= k <= X.cols().");
        }
        if (X.rows() > X.cols() && X.size() > 0) { // X.size() > 0 to allow empty matrices if k=0
            throw std::invalid_argument("Input matrix X must have at least as many columns as rows.");
        }

        std::vector<Eigen::Index> cols(static_cast<size_t>(k)); // Safe cast for vector size
        for (Eigen::Index i = 0; i < k; ++i) {
            cols[static_cast<size_t>(i)] = i;
        }
        return cols;
    }

    /*!
     * @brief Calculates a theoretical lower bound related to the selected submatrix.
     * @tparam norm The type of matrix norm to consider for the bound calculation
     *              (must be `Norm::Frobenius` or `Norm::Spectral`).
     * @param X The input `m x n` matrix.
     * @param k The number of columns that would be selected.
     * @return A scalar value representing the calculated bound.
     *
     * The bound is often interpreted as a lower limit on a ratio like
     * `||X_pinv||^2 / ||X_S_pinv||^2`, where `X_pinv` is the pseudoinverse of `X`
     * and `X_S_pinv` is the pseudoinverse of the submatrix formed by selected columns.
     * The exact formula and interpretation depend on the theoretical underpinnings.
     * This version takes the full matrix `X` as input.
     *
     * @note This method relies on `static_assert` to ensure `norm` is supported.
     */
    template <Norm norm>
    scalar bound(const Eigen::MatrixX<scalar>& X, Eigen::Index k) const {
        static_assert(norm == Norm::Frobenius || norm == Norm::Spectral,
                      "SelectorBase::bound: This norm is unsupported for the bound calculation!");
        // Basic validation
        if (k < 0 || k > X.cols() || (X.rows() > 0 && k < X.rows())) {
             throw std::invalid_argument("Invalid k for bound calculation.");
        }
        return boundInternal(X.rows(), X.cols(), k, norm);
    }

    /*!
     * @brief Calculates a theoretical lower bound based on matrix dimensions.
     * @tparam norm The type of matrix norm to consider for the bound calculation
     *              (must be `Norm::Frobenius` or `Norm::Spectral`).
     * @param m The number of rows in the matrix.
     * @param n The number of columns in the matrix.
     * @param k The number of columns that would be selected.
     * @return A scalar value representing the calculated bound.
     *
     * This overloaded version calculates the bound based purely on the dimensions
     * `m`, `n`, and `k`, and the chosen `norm`. It does not require the matrix data itself.
     *
     * @note This method relies on `static_assert` to ensure `norm` is supported.
     */
    template <Norm norm>
    scalar bound(Eigen::Index m, Eigen::Index n, Eigen::Index k) const {
        static_assert(norm == Norm::Frobenius || norm == Norm::Spectral,
                      "SelectorBase::bound: This norm is unsupported for the bound calculation!");
        // Basic validation
        if (m < 0 || n < 0 || k < 0 || k > n || (m > 0 && k < m) || m > n) {
             throw std::invalid_argument("Invalid dimensions (m, n, k) for bound calculation.");
        }
        return boundInternal(m, n, k, norm);
    }

  private:
    /*!
     * @brief Internal implementation for calculating the bound.
     * @param m Number of rows.
     * @param n Number of columns.
     * @param k Number of selected columns.
     * @param norm The norm type.
     * @return The calculated bound value.
     *
     * This virtual method allows derived classes to potentially provide
     * more specific bound calculations if the default (0) is not appropriate
     * or if they have information to compute a tighter bound.
     * The default implementation returns 0.
     * @note This is private and virtual, intended for internal polymorphic behavior.
     *       Consider making it protected if derived classes are meant to call it directly
     *       with different parameters, but for now, it's an internal detail of the public `bound` methods.
     */
    scalar boundInternal(Eigen::Index m, Eigen::Index n, Eigen::Index k,
                                 Norm norm) const {
        // Suppress unused parameter warnings if k is not used in a more complex default bound.
        // These parameters are available for derived classes that might override this.
        (void)m;
        (void)n;
        (void)k;
        (void)norm;
        return static_cast<scalar>(0);
    }
};

} // namespace MatSubset

#endif // MAT_SUBSET_SELECTOR_BASE_H