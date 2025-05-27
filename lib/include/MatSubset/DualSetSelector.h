#ifndef MAT_SUBSET_DUAL_SET_SELECTOR_H
#define MAT_SUBSET_DUAL_SET_SELECTOR_H

#include <cmath> // For std::sqrt, std::pow

#include <Eigen/Eigenvalues> // For Eigen::SelfAdjointEigenSolver
#include <Eigen/SVD>         // For Eigen::BDCSVD

#include "SelectorBase.h" // Base class

namespace MatSubset {

/*!
 * @brief Approximates subset selection problem for matrices using
 * the dual set algorithm (Algorithm 3 from Avron and Boutsidis, 2012).
 * @tparam Scalar The underlying scalar type (e.g., `float`, `double`).
 *
 * This class implements Algorithm 3 ("A deterministic greedy selection
 * algorithm for subset selection (Theorem 3.5.)") from Avron and Boutsidis
 * (2012), "Faster Subset Selection for Matrices and Applications". This
 * algorithm is based on the dual set spectral sparsification framework by
 * Batson, Spielman, and Srivastava.
 *
 * The algorithm selects columns that receive non-zero weights after \f$ k \f$
 * iterations. It is designed to produce a column-submatrix \f$ X_S \f$ whose
 * pseudoinverse norm is controlled.
 *
 * @note This algorithm requires \f$ k > m \f$.
 */
template <typename Scalar> class DualSetSelector : public SelectorBase<Scalar> {
  public:
    /*!
     * @brief Default constructor for `DualSetSelector`.
     */
    DualSetSelector() = default;

    /*!
     * @brief Gets the human-readable name of the algorithm.
     * @return The string "dual set".
     */
    std::string getAlgorithmName() const override { return "dual set"; }

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
        // Parameters from SelectorBase: X (m_rows x n_cols matrix), k (columns
        // to select)
        const Eigen::Index m = X.rows(); // m_rows
        const Eigen::Index n = X.cols(); // n_cols

        // dual set algorithm (Lemma 3.4 in Avron & Boutsidis, from BSS)
        // requires k > m (rows) for delta_u initialization.
        assert(k > m && "dual set algorithm requires k > m (rows).");

        Eigen::BDCSVD<Eigen::MatrixX<Scalar>> svd(X, Eigen::ComputeThinV);
        Eigen::MatrixX<Scalar> V = svd.matrixV().transpose(); // V is m x n
        Eigen::MatrixX<Scalar> A = Eigen::MatrixX<Scalar>::Zero(m, m);
        Eigen::VectorX<Scalar> s = Eigen::VectorX<Scalar>::Zero(n);

        Scalar delta_l = static_cast<Scalar>(1.0);
        Scalar l = -std::sqrt(static_cast<Scalar>(k * m));

        Scalar sqrt_k_val = std::sqrt(static_cast<Scalar>(k));
        Scalar sqrt_m_val = std::sqrt(static_cast<Scalar>(m));
        Scalar delta_u_denominator = sqrt_k_val - sqrt_m_val;
        // Assertion k > m ensures delta_u_denominator > 0

        Scalar delta_u = (std::sqrt(static_cast<Scalar>(n)) + sqrt_k_val) /
                         delta_u_denominator;
        Scalar u = delta_u * std::sqrt(static_cast<Scalar>(k * n));

        for (Eigen::Index i = 0; i < k; ++i) {
            Eigen::ArrayX<Scalar> L_vals = calculateL(V, delta_l, A, l);
            Eigen::ArrayX<Scalar> U_vals = calculateU(delta_u, s.array(), u);

            l += delta_l;
            u += delta_u;

            Eigen::Index max_idx;
            (L_vals - U_vals).maxCoeff(&max_idx);

            Scalar t =
                static_cast<Scalar>(2.0) / (L_vals(max_idx) + U_vals(max_idx));

            s(max_idx) += t;
            A += t * V.col(max_idx) * V.col(max_idx).transpose();
        }

        std::vector<Eigen::Index> indices;
        indices.reserve(k);
        for (Eigen::Index i = 0; i < s.size(); ++i) {
            if (s(i) > static_cast<Scalar>(0)) {
                indices.push_back(i);
            }
        }

        // dual set guarantees at most k non-zero weights s_i.
        // If strictly fewer than k are found, padding may be desired to meet
        // the contract of returning k indices.
        if (indices.size() < static_cast<size_t>(k)) {
            std::vector<bool> is_already_selected(n, false);
            for (Eigen::Index selected_idx : indices) {
                is_already_selected[static_cast<size_t>(selected_idx)] = true;
            }

            for (Eigen::Index i_fill = 0;
                 i_fill < n && indices.size() < static_cast<size_t>(k);
                 ++i_fill) {
                if (!is_already_selected[static_cast<size_t>(i_fill)]) {
                    indices.push_back(i_fill);
                }
            }
        }
        // Ensure exactly k 
        if (indices.size() > static_cast<size_t>(k)) {
            indices.resize(static_cast<size_t>(k));
        }

        return indices;
    }

  protected:
     /*!
     * @brief Calculates the theoretical bound for the dual set selection
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
     * The bound is calculated based on the Theorem 3.5 in Avron and Boutsidis
     * (2012).
     */
    Scalar boundImpl(Eigen::Index m, Eigen::Index n, Eigen::Index k,
                     Norm norm) const override {

        assert(k > m && "dual set algorithm requires k > m (rows).");

        Scalar sqrt_k_plus_1 = std::sqrt(static_cast<Scalar>(k + 1));
        Scalar sqrt_m_val = std::sqrt(static_cast<Scalar>(m)); // m is rows
        Scalar sqrt_n_val = std::sqrt(static_cast<Scalar>(n)); // n is cols

        Scalar numerator = sqrt_k_plus_1 - sqrt_m_val;
        Scalar denominator = sqrt_n_val + sqrt_k_plus_1;

        Scalar ratio = numerator / denominator;
        return std::pow(ratio, 2);
    }

  private:
    Eigen::ArrayX<Scalar> calculateL(const Eigen::MatrixX<Scalar> &V,
                                     Scalar delta_l,
                                     const Eigen::MatrixX<Scalar> &A,
                                     Scalar l) const {

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<Scalar>> decomposition(A);
        Eigen::MatrixX<Scalar> U_eigenvecs = decomposition.eigenvectors();
        Eigen::ArrayX<Scalar> S_eigenvals = decomposition.eigenvalues().array();

        Eigen::ArrayX<Scalar> D_vals = (S_eigenvals - (l + delta_l)).inverse();
        Eigen::MatrixX<Scalar> M_1 = D_vals.matrix().asDiagonal();

        Eigen::ArrayX<Scalar> term_S_l_inv = (S_eigenvals - l).inverse();
        Scalar M2_denominator = D_vals.sum() - term_S_l_inv.sum();

        Eigen::MatrixX<Scalar> M_2 =
            (D_vals.square() / M2_denominator).matrix().asDiagonal();
        return (V.transpose() * U_eigenvecs * (M_2 - M_1) *
                U_eigenvecs.transpose() * V)
            .diagonal();
    }

    Eigen::ArrayX<Scalar>
    calculateU(Scalar delta_u, const Eigen::ArrayX<Scalar> &s, Scalar u) const {

        Eigen::ArrayX<Scalar> term1_inv = ((u + delta_u) - s).inverse();

        Eigen::ArrayX<Scalar> term_u_s_inv = (u - s).inverse();
        Scalar term2_denominator = term_u_s_inv.sum() - term1_inv.sum();

        return term1_inv + term1_inv.square() / term2_denominator;
    }
};

} // namespace MatSubset

#endif // MAT_SUBSET_DUAL_SET_SELECTOR_H