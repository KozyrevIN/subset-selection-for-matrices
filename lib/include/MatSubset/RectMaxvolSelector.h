#ifndef MAT_SUBSET_RECT_MAXVOL_SELECTOR_H
#define MAT_SUBSET_RECT_MAXVOL_SELECTOR_H

#include <Eigen/QR> // For Eigen::CompleteOrthogonalDecomposition

#include "DominantSelector.h" // Base class
#include "Enums.h"            // For MatSubset::Norm

namespace MatSubset {

/*!
 * @brief Approximates subset selection problem for matrices by iteratively
 * selecting columns that maximize volume of the submatrix.
 * @tparam Scalar The underlying scalar type (e.g., `float`, `double`).
 *
 * This class implements Algorithm 1 from Michalev and Oseledets (2018)
 * "Rectangular maximum-volume submatrices and their applications". This
 * algorithm selects the initial subset of size \f$m\f$ using Maxvol algorithm
 * (equivalent to Dominant in case \f$m = k\f$) and then iteratively adds
 * columns to it in order to maximize the volume of the submatrix.
 */
template <typename Scalar>
class RectMaxvolSelector : public DominantSelector<Scalar> {
  public:
    /*!
     * @brief Default constructor for `RectMaxvolSelector`.
     */
    explicit RectMaxvolSelector(Scalar c) : DominantSelector<Scalar>(c){};

    /*!
     * @brief Gets the human-readable name of the algorithm.
     * @return The string "rect-maxvol".
     */
    std::string getAlgorithmName() const override { return "rect-maxvol"; }

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
                                               Eigen::Index k,
                                               Eigen::Index *swap_count) override {

        const Eigen::Index m = X.rows();
        const Eigen::Index n = X.cols();

        // Preparing starting sets of indices
        std::vector<Eigen::Index> selected_indices =
            DominantSelector<Scalar>::selectSubsetImpl(X, m, nullptr);
        std::vector<Eigen::Index> remaining_indices;
        remaining_indices.reserve(n - m);

        std::vector<bool> is_already_selected(n, false);
        for (Eigen::Index i : selected_indices) {
            is_already_selected[static_cast<size_t>(i)] = true;
        }

        for (Eigen::Index i = 0; i < n; ++i) {
            if (!is_already_selected[static_cast<size_t>(i)]) {
                remaining_indices.push_back(i);
            }
        }

        // Initializing necessary matrices
        Eigen::MatrixX<Scalar> X_selected = X(Eigen::all, selected_indices);
        Eigen::MatrixX<Scalar> X_remaining = X(Eigen::all, remaining_indices);
        Eigen::MatrixX<Scalar> X_selected_dag =
            X_selected.completeOrthogonalDecomposition().pseudoInverse();

        // Each added column makes the selected set one wider, so
        // \f$ X_{\mathcal{S}}^{\dag} \f$ gains a row and so does
        // \f$ C = X_{\mathcal{S}}^{\dag} X_{\mathcal{R}} \f$. That growing row
        // is part of the algorithm, not bookkeeping: the next pivot's scores
        // are inner products of *full* columns of C, so a C that stayed m rows
        // tall would score every later candidate on a truncated vector and the
        // selection would stop improving. The final shape is known up front, so
        // allocate it once and work in the live top-left corner.
        Eigen::MatrixX<Scalar> C_store(k, n - m);
        C_store.topRows(m).noalias() = X_selected_dag * X_remaining;

        Eigen::Index rows = m;
        Eigen::Index active = n - m;

        Eigen::ArrayX<Scalar> l =
            C_store.block(0, 0, rows, active).colwise().squaredNorm();
        Eigen::Index j_max;
        Scalar max_val = l.maxCoeff(&j_max);

        // Scratch, allocated once. The pivot column is copied out because the
        // rank-1 update overwrites the very column it reads.
        Eigen::VectorX<Scalar> pivot_col(k);
        Eigen::RowVectorX<Scalar> v(n - m);

        // Main loop
        for (Eigen::Index i = m; i < k; ++i) {

            auto C = C_store.block(0, 0, rows, active);
            auto v_a = v.head(active);
            auto pivot = pivot_col.head(rows);

            pivot = C.col(j_max);
            const Scalar denom = static_cast<Scalar>(1) + max_val;

            // v_j = c_p . c_j, unscaled: the score update is
            // l_j <- l_j - v_j^2 / (1 + l_p), and that single factor of
            // (1 + l_p) is exactly what makes l_j remain the squared norm of
            // the grown column, appended row included.
            v_a.noalias() = pivot.transpose() * C;
            l -= v_a.transpose().array().abs2() / denom;

            // v_a becomes the appended row; the body update uses the same
            // scaled row, so scale in place and use it twice.
            v_a /= denom;
            C.noalias() -= pivot * v_a;
            C_store.block(rows, 0, 1, active) = v_a;
            ++rows;

            // Add Index j_max to selected ones
            selected_indices.push_back(
                remaining_indices[static_cast<size_t>(j_max)]);

            Eigen::Index new_size = active - 1;
            if (j_max < new_size) {
                remaining_indices[static_cast<size_t>(j_max)] =
                    remaining_indices[static_cast<size_t>(new_size)];
                l(j_max) = l(new_size);
                C_store.block(0, j_max, rows, 1) =
                    C_store.block(0, new_size, rows, 1);
            }

            remaining_indices.resize(static_cast<size_t>(new_size));
            l.conservativeResize(new_size);
            active = new_size;

            // Select new index to append
            max_val = l.maxCoeff(&j_max);
        }

        return selected_indices;
    }

    /*!
     * @brief Calculates the theoretical bound for the RectMaxvol selection
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
     * No guarantees for Rect-maxvol algorithm are proven except for the bound
     * of the underlying Maxvol algorithm, which is equivalent to Dominant
     * algorithm for \f$m = k\f$.
     *
     * @note The bound depends on the parameter c.
     * @note The bound for Frobenius norm from Osinsky (2024),
     * Corollary 1, is not standard (it mixes different norms), we use
     * inequality \f$ \lVert X \rVert_2 \le \lVert X \rVert_F \f$ to produce a
     * looser bound, which fits into our framework.
     */
    Scalar boundImpl(Eigen::Index m, Eigen::Index n, Eigen::Index k,
                     Norm norm) const override {

        return DominantSelector<Scalar>::boundImpl(m, n, m, norm);
    }
};

} // namespace MatSubset

#endif // MAT_SUBSET_RECT_MAXVOL_SELECTOR_H