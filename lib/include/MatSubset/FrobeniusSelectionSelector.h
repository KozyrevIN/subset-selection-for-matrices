#ifndef MAT_SUBSET_FROBENIUS_SELECTION_SELECTOR_H
#define MAT_SUBSET_FROBENIUS_SELECTION_SELECTOR_H

#include <Eigen/QR> // For Eigen::HouseholderQR

#include "FrobeniusPivotingBase.h" // Base class

namespace MatSubset {

/*!
 * @brief Approximates subset selection problem for matrices by iteratively
 * selecting columns that maximize the Frobenius norm of the pseudoinverse of
 * the selected submatrix.
 * @tparam Scalar The underlying scalar type (e.g., `float`, `double`).
 *
 * This algorithm operates on the trnsposed right singular vectors of the input
 * matrix. It starts with a highly nondegenerate \f$ m \times m \f$ submatrix
 * obtained from `FrobeniusPivotingSelector`, then greedily selects additional
 * columns that minimize the frobenius norm of the pseudoinverse of selected
 * submatrix using efficient rank-1 updates.
 */
template <typename Scalar>
class FrobeniusSelectionSelector : public FrobeniusPivotingBase<Scalar> {
  public:
    /*!
     * @brief Default constructor for `FrobeniusSelectionSelector`.
     */
    FrobeniusSelectionSelector() = default;

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
     * @param k The number of columns to select. Must satisfy \f$ k \geq m \f$.
     * @return A `std::vector` of `Eigen::Index` containing the 0-based indices
     * of the selected columns.
     */
    std::vector<Eigen::Index>
    selectSubsetImpl(const Eigen::MatrixX<Scalar> &X, Eigen::Index k,
                     Eigen::Index *swap_count) override {

        const Eigen::Index m = X.rows();
        const Eigen::Index n = X.cols();

        // LQ decomposition
        Eigen::HouseholderQR<Eigen::MatrixX<Scalar>> qr(X.transpose());
        Eien::MatrixX<Scalar> V =
            (qr.householderQ() * Eigen::MatrixX<Scalar>::Identity(n, m))
                .transpose();

        Eigen::MatrixX<Scalar> Wt; // W = V_S^{-1} V, transposed (n x m)
        std::vector<Eigen::Index> indices =
            FrobeniusPivotingBase<Scalar>::selectStartingSet(V, &Wt);

        if (k == m) {
            indices.resize(k);
            return indices;
        }

        std::vector<Eigen::Index> selected_indices(indices.begin(),
                                                   indices.begin() + m);
        selected_indices.reserve(k);
        std::vector<Eigen::Index> remaining_indices(indices.begin() + m,
                                                    indices.end());

        Eigen::MatrixX<Scalar> V_remaining = V.rightCols(n - m);

        // The starting set hands back W = V_S^{-1} V with V_S = V.leftCols(m),
        // upper triangular after its Householder sweep. Since
        // M = (V_S V_S^T)^{-1} = B^T B with B = V_S^{-1}, the greedy scores
        // initialize directly from W:
        //   d_j = 1 + v_j^T M v_j = 1 + ||W_j||^2,
        //   l_j = ||M v_j||^2 = ||B^T W_j||^2.
        // The triangular inverse is also better conditioned than inverting
        // the Gram matrix V_S V_S^T, which squares the condition number.
        Eigen::MatrixX<Scalar> B =
            V.leftCols(m).template triangularView<Eigen::Upper>().solve(
                Eigen::MatrixX<Scalar>::Identity(m, m));
        Eigen::MatrixX<Scalar> M(m, m);
        M.noalias() = B.transpose() * B;

        // Row j of Wt.bottomRows(n - m) is W_j of remaining column j; row
        // norms are accumulated column-by-column (contiguous slices).
        Eigen::ArrayX<Scalar> d = Eigen::ArrayX<Scalar>::Ones(n - m);
        for (Eigen::Index c = 0; c < m; ++c) {
            d += Wt.col(c).tail(n - m).array().square();
        }
        // Row j of Tt is (B^T W_j)^T.
        Eigen::MatrixX<Scalar> Tt(n - m, m);
        Tt.noalias() = Wt.bottomRows(n - m) * B;
        Eigen::ArrayX<Scalar> l = Eigen::ArrayX<Scalar>::Zero(n - m);
        for (Eigen::Index c = 0; c < m; ++c) {
            l += Tt.col(c).array().square();
        }

        // The pool shrinks by swap-with-last; `r` is the active width and the
        // matrices keep their allocation. The projection buffers are
        // preallocated at the full width and used through .head(r).
        Eigen::Index r = n - m;
        Eigen::VectorX<Scalar> vec_1(m);
        Eigen::VectorX<Scalar> vec_2(m);
        Eigen::ArrayX<Scalar> mul_1(r);
        Eigen::ArrayX<Scalar> mul_2(r);
        for (Eigen::Index i = m; i < k; ++i) {
            Eigen::Index j_max;
            (l.head(r) / d.head(r)).maxCoeff(&j_max);

            Eigen::VectorX<Scalar> v = V_remaining.col(j_max);
            Scalar denom = d(j_max);

            selected_indices.push_back(
                remaining_indices[static_cast<size_t>(j_max)]);
            --r;
            if (j_max < r) {
                remaining_indices[static_cast<size_t>(j_max)] =
                    remaining_indices[static_cast<size_t>(r)];
                l(j_max) = l(r);
                d(j_max) = d(r);
                V_remaining.col(j_max) = V_remaining.col(r);
            }
            remaining_indices.resize(static_cast<size_t>(r));

            vec_1.noalias() = M * v;
            vec_2.noalias() = M * vec_1;
            Scalar mul_3 = vec_2.dot(v);

            mul_1.head(r).matrix().noalias() =
                V_remaining.leftCols(r).transpose() * vec_1;
            mul_2.head(r).matrix().noalias() =
                V_remaining.leftCols(r).transpose() * vec_2;

            M.noalias() -= vec_1 * vec_1.transpose() / denom;

            d.head(r) -= mul_1.head(r).square() / denom;
            mul_1.head(r) /= denom;
            l.head(r) +=
                mul_1.head(r) * (mul_1.head(r) * mul_3 - 2 * mul_2.head(r));
        }

        return selected_indices;
    }

    /*!
     * @brief Calculates the theoretical bound for Frobenius selection
     * algorithm.
     * @param m The number of rows in the matrix.
     * @param n The number of columns in the matrix.
     * @param k The number of columns that would be selected.
     * @param norm The type of matrix norm (`Norm::Frobenius` or
     * `Norm::Spectral`).
     * @return A `Scalar` value representing the calculated lower bound on the
     * ratio \f$ \lVert X^{\dag} \rVert^{2}/\lVert X_{\mathcal{S}}^{\dag}
     * \rVert^{2} \f$.
     */
    Scalar boundImpl(Eigen::Index m, Eigen::Index n, Eigen::Index k,
                     Norm norm) const override {

        Scalar S_n = static_cast<Scalar>(n);
        Scalar S_m = static_cast<Scalar>(m);
        Scalar S_k = static_cast<Scalar>(k);
        Scalar bound_val = S_k / (S_m * (S_n - S_m + 1));
        if (norm == Norm::Spectral) {
            bound_val /= static_cast<Scalar>(m);
        }
        return bound_val;
    }
};

} // namespace MatSubset

#endif // MAT_SUBSET_FROBENIUS_SELECTION_SELECTOR_H