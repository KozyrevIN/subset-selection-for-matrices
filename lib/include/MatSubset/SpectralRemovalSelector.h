#ifndef MAT_SUBSET_SPECTRAL_REMOVAL_H
#define MAT_SUBSET_SPECTRAL_REMOVAL_H

#include <Eigen/SVD> // For Eigen::BDCSVD

#include "FrobeniusRemovalSelector.h" // Base class for this selector

namespace MatSubset {

/*!
 * @brief Approximates subset selection problem for matrices using a
 * spectral norm-based greedy removal strategy.
 * @tparam Scalar The underlying scalar type (e.g., `float`, `double`).
 *
 * This class implements Algorithm 2 ("Deterministic Greedy Removal (spectral
 * norm)") from Avron and Boutsidis (2012), "Faster Subset Selection for
 * Matrices and Applications".
 *
 * It adapts the `FrobeniusRemovalSelector` by running its core logic on the
 * transpose of the right singular vectors (\f$ V^T \f$) of the input matrix \f$
 * X \f$. This slightly improves resulting spectral norm bounds, but worsens
 * Frobenius norm bounds.
 */
template <typename Scalar>
class SpectralRemovalSelector : public FrobeniusRemovalSelector<Scalar> {
  public:
    /*!
     * @brief Constructor for `SpectralRemovalSelector`.
     * @param eps Small positive tolerance value passed to the underlying
     *            `FrobeniusRemovalSelector`. Defaults to \f$ 1e-6 \f$.
     */
    explicit SpectralRemovalSelector(Scalar eps = static_cast<Scalar>(1e-6))
        : FrobeniusRemovalSelector<Scalar>(eps) {}

    /*!
     * @brief Gets the human-readable name of the algorithm.
     * @return The string "spectral removal".
     */
    std::string getAlgorithmName() const override { return "spectral removal"; }

  protected:
    /*!
     * @brief Core implementation for selecting a subset of \f$ k \f$ columns.
     * @param X The input matrix (dimensions \f$ m \times n \f$) from which
     * columns are to be selected. It is assumed that \f$ X \f$ is full rank
     * for theoretical guarantees.
     * @param k The number of columns to select.
     * @return A `std::vector` of `Eigen::Index` containing the 0-based indices
     * of the selected columns.
     */
    std::vector<Eigen::Index> selectSubsetImpl(const Eigen::MatrixX<Scalar> &X,
                                               Eigen::Index k) override {
        Eigen::BDCSVD<Eigen::MatrixX<Scalar>> svd(X, Eigen::ComputeThinV);
        Eigen::MatrixX<Scalar> V = svd.matrixV().transpose();

        return FrobeniusRemovalSelector<Scalar>::selectSubsetImpl(V, k);
    }

    /*!
     * @brief Calculates theoretical lower bounds for the spectral removal
     * strategy.
     * @param m The number of rows in the matrix.
     * @param n The number of columns in the matrix.
     * @param k The number of columns that would be selected.
     * @param norm The type of matrix norm (`Norm::Frobenius` or
     * `Norm::Spectral`).
     * @return A `Scalar` value representing the calculated lower bound on the
     * ratio \f$ \lVert X^{\dag} \rVert^{2}/\lVert X_{\mathcal{S}}^{\dag}
     * \rVert^{2} \f$.
     *
     * @note The bound for Frobenius norm from Avron and Boutsidis (2012),
     * Corollary 3.3, is not standard (it mixes different norms), we use
     * inequality \f$ \lVert X \rVert_2 \le \lVert X \rVert_F \f$ to produce a
     * looser bound, wich fits into our framework.
     */
    Scalar boundImpl(Eigen::Index m, Eigen::Index n, Eigen::Index k,
                     Norm norm) const override {

        Scalar bound_val = static_cast<Scalar>(0);
        Scalar diff_k_m = static_cast<Scalar>(k - m + 1);

        if (norm == Norm::Spectral) {

            Scalar numerator = static_cast<Scalar>(m * (n - k));
            bound_val = static_cast<Scalar>(1) /
                        (static_cast<Scalar>(1) + numerator / diff_k_m);

        } else if (norm == Norm::Frobenius) {

            bound_val = diff_k_m / static_cast<Scalar>(n - m + 1) /
                        static_cast<Scalar>(m);
        }

        return bound_val;
    }
};

} // namespace MatSubset

#endif // MAT_SUBSET_SPECTRAL_REMOVAL_H