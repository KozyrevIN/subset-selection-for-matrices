#ifndef MAT_SUBSET_FROBENIUS_SELECTION_SELECTOR_H
#define MAT_SUBSET_FROBENIUS_SELECTION_SELECTOR_H

#include <Eigen/SVD> // For Eigen::BDCSVD

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
    std::vector<Eigen::Index> selectSubsetImpl(const Eigen::MatrixX<Scalar> &X,
                                               Eigen::Index k,
                                               Eigen::Index *swap_count) override {

        const Eigen::Index m = X.rows();
        const Eigen::Index n = X.cols();

        Eigen::BDCSVD<Eigen::MatrixX<Scalar>> svd(X, Eigen::ComputeThinV);
        Eigen::MatrixX<Scalar> V = svd.matrixV().transpose();

        std::vector<Eigen::Index> indices =
            FrobeniusPivotingBase<Scalar>::selectStartingSet(V);

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
        Eigen::MatrixX<Scalar> M =
            (V.leftCols(m) * V.leftCols(m).transpose()).inverse();
        Eigen::ArrayX<Scalar> l =
            (V_remaining.transpose() * (M * M) * V_remaining).diagonal();
        Eigen::ArrayX<Scalar> d =
            static_cast<Scalar>(1) +
            (V_remaining.transpose() * M * V_remaining).diagonal().array();

        for (Eigen::Index i = m; i < k; ++i) {
            Eigen::ArrayX<Scalar> scores = l / d;
            Eigen::Index j_max;
            scores.maxCoeff(&j_max);

            Eigen::VectorX<Scalar> v = V_remaining.col(j_max);
            Scalar denom = d(j_max);

            addColumn(selected_indices, remaining_indices, l, d, V_remaining,
                      j_max);

            Eigen::VectorX<Scalar> vec_1 = M * v;
            Eigen::VectorX<Scalar> vec_2 = M * vec_1;
            Eigen::ArrayX<Scalar> mul_1 = vec_1.transpose() * V_remaining;
            Eigen::ArrayX<Scalar> mul_2 = vec_2.transpose() * V_remaining;
            Scalar mul_3 = (vec_2.transpose() * v).value();

            M -= vec_1 * vec_1.transpose() / denom;

            d -= mul_1.square() / denom;
            mul_1 /= denom;
            l += mul_1 * (mul_1 * mul_3 - 2 * mul_2);
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

  private:
    /*!
     * @brief Helper to move column `j_selected` from remaining set to selected
     * set. Modifies all parameters in place by copying the last remaining
     * element to `j_selected` and then resizing.
     * @param selected_indices Vector of selected column indices (grows by 1).
     * @param remaining_indices Vector of remaining column indices (shrinks by
     * 1).
     * @param l Array of l-scores (numerators for the selection criterion).
     * @param d Array of d-scores (denominators for the selection criterion).
     * @param V_remaining Matrix of remaining \f$ V \f$ columns (from SVD).
     * @param j_selected The 0-based index *within the current remaining set* to
     * select and move.
     */
    void addColumn(std::vector<Eigen::Index> &selected_indices,
                   std::vector<Eigen::Index> &remaining_indices,
                   Eigen::ArrayX<Scalar> &l, Eigen::ArrayX<Scalar> &d,
                   Eigen::MatrixX<Scalar> &V_remaining,
                   Eigen::Index j_selected) const {

        selected_indices.push_back(
            remaining_indices[static_cast<size_t>(j_selected)]);

        Eigen::Index new_size =
            static_cast<Eigen::Index>(remaining_indices.size()) - 1;
        if (j_selected < new_size) {
            remaining_indices[static_cast<size_t>(j_selected)] =
                remaining_indices[static_cast<size_t>(new_size)];
            l(j_selected) = l(new_size);
            d(j_selected) = d(new_size);
            V_remaining.col(j_selected) = V_remaining.col(new_size);
        }

        remaining_indices.resize(static_cast<size_t>(new_size));
        l.conservativeResize(new_size);
        d.conservativeResize(new_size);
        V_remaining.conservativeResize(Eigen::NoChange, new_size);
    }
};

} // namespace MatSubset

#endif // MAT_SUBSET_FROBENIUS_SELECTION_SELECTOR_H