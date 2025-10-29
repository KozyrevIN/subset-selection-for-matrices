#ifndef MAT_SUBSET_FROBENIUS_SELECTION_SELECTOR_H
#define MAT_SUBSET_FROBENIUS_SELECTION_SELECTOR_H

#include <Eigen/QR>  // For Eigen::CompleteOrthogonalDecomposition
#include <Eigen/SVD> // For Eigen::BDCSVD

#include "FrobeniusPivotingBase.h" // Base class

namespace MatSubset {

/*!
 * @brief Approximates subset selection problem for matrices iteratively
 * selecting columns that maximize the Frobenius norm of the pseudoinverse of
 * the selected submatrix.
 * @tparam Scalar The underlying scalar type (e.g., `float`, `double`).
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
     * @param k The number of columns to select. Must be equal to \f$ m \f$ for
     * this method.
     * @return A `std::vector` of `Eigen::Index` containing the 0-based indices
     * of the selected columns.
     */
    std::vector<Eigen::Index> selectSubsetImpl(const Eigen::MatrixX<Scalar> &X,
                                               Eigen::Index k) override {

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
            
            /*
            d -= mul_1.square() / denom;
            mul_1 /= denom;
            l += mul_1 * (mul_1 * mul_3 - 2 * mul_2);
            */

            l = (V_remaining.transpose() * (M * M) * V_remaining).diagonal();
            d = static_cast<Scalar>(1) +
                (V_remaining.transpose() * M * V_remaining).diagonal().array();
        }

        return selected_indices;
    }

  private:
    /*!
     * @brief Helper to remove column `idx_to_remove` from active data
     * structures. Modifies all parameters in place by copying the last active
     * element to `idx_to_remove` and then resizing.
     * @param cols Vector of original column indices.
     * @param l Array of l-scores (numerators for the removal criterion).
     * @param d Array of d-scores (denominators for the removal
     * criterion).
     * @param V Matrix of active \f$ V \f$ columns (from SVD).
     * @param V_dag Matrix of active \f$ V^{\dag} \f$ (pseudoinverse related)
     * columns.
     * @param idx_to_remove The 0-based index *within the current active set* to
     * remove.
     */
    void addColumn(std::vector<Eigen::Index> &selected_indices,
                   std::vector<Eigen::Index> &remaining_indices,
                   Eigen::ArrayX<Scalar> &l, Eigen::ArrayX<Scalar> &d,
                   Eigen::MatrixX<Scalar> &V_remaining,
                   Eigen::Index j_selected) const {

        Eigen::Index new_size =
            static_cast<Eigen::Index>(remaining_indices.size()) - 1;

        if (j_selected < new_size) {
            selected_indices.push_back(j_selected);
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