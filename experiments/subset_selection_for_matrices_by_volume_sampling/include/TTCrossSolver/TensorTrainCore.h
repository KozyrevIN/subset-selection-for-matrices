#ifndef MAT_SUBSET_EXPERIMENTS_TENSOR_TRAIN_CORE_H
#define MAT_SUBSET_EXPERIMENTS_TENSOR_TRAIN_CORE_H

#include <algorithm> // For std::max
#include <cassert>   // For assert
#include <memory>    // For std::unique_ptr
#include <utility>   // For std::pair, std::move
#include <vector>    // For std::vector

#include <Eigen/Core> // For Eigen::MatrixX, Eigen::Map, Eigen::Index
#include <Eigen/QR>   // For Eigen::ColPivHouseholderQR
#include <Eigen/SVD>  // For Eigen::BDCSVD

#include <MatSubset/MatSubset.h> // For subset selection algorithms and utils

namespace MatSubset::Experiments {

/*!
 * @brief A single TT-core of a tensor-train, logically of shape
 * \f$ r_0 \times n \times r_1 \f$ (left-rank, mode, right-rank).
 * @tparam Scalar The underlying scalar type (e.g. `float`, `double`).
 *
 * The core is stored as its left unfolding, a column-major matrix of shape
 * \f$ (r_0 n) \times r_1 \f$. The orthogonalization sweeps used by TT-cross
 * absorb an incoming \f$ R \f$ factor from the neighbouring core,
 * re-orthogonalize the corresponding unfolding via QR, overwrite the stored
 * unfolding with the orthonormal factor, and return the new \f$ R \f$ factor to
 * be carried into the next core.
 *
 * The SVD sweeps additionally truncate the rank: they absorb \f$ R \f$, compute
 * \f$ U \Sigma V^{\top} \f$, drop the trailing singular values whose Frobenius
 * energy stays within the tolerance, overwrite the unfolding with the
 * orthonormal factor, and return the remaining factor to carry to the
 * neighbour.
 */
template <typename Scalar> class TensorTrainCore {

  public:
    /*!
     * @brief Constructs a zero-initialized core of shape r0 x n x r1.
     */
    TensorTrainCore(Eigen::Index r0, Eigen::Index n, Eigen::Index r1)
        : left_rank(r0), mode(n), right_rank(r1),
          left_unfolding(Eigen::MatrixX<Scalar>::Zero(r0 * n, r1)) {}

    /*!
     * @brief Constructs a core from an existing left unfolding.
     * @param unfolding Left unfolding of shape (r0 * n) x r1.
     * @param n The mode size, needed to recover r0 from unfolding.rows().
     */
    TensorTrainCore(Eigen::MatrixX<Scalar> unfolding, Eigen::Index n)
        : left_rank(unfolding.rows() / n), mode(n),
          right_rank(unfolding.cols()), left_unfolding(std::move(unfolding)) {}

    [[nodiscard]] Eigen::Index leftRank() const { return left_rank; }
    [[nodiscard]] Eigen::Index modeSize() const { return mode; }
    [[nodiscard]] Eigen::Index rightRank() const { return right_rank; }

    /*!
     * @brief Sets the raw left unfolding, shape (r0 * n) x r1.
     */
    void setLeftUnfolding(const Eigen::MatrixX<Scalar> &unfolding) {
        left_unfolding = unfolding;
    }

    /*!
     * @brief The raw left unfolding, shape (r0 * n) x r1.
     */
    [[nodiscard]] const Eigen::MatrixX<Scalar> &leftUnfolding() const {
        return left_unfolding;
    }

    /*!
     * @brief The mode-`i` slice of the core, the matrix \f$ G[:, i, :] \f$ of
     * shape r0 x r1.
     * @param i The mode index, in [0, n).
     *
     * In the left unfolding (r0 * n) x r1, entry \f$ G[a, i, c] \f$ lives at row
     * \f$ a + r_0 i \f$, so the slice is the contiguous row block
     * \f$ [i r_0, (i + 1) r_0) \f$. Useful for evaluating the train at fibers:
     * fixing every mode index reduces each core to one such matrix and the
     * tensor entry is their product.
     */
    [[nodiscard]] Eigen::MatrixX<Scalar> modeSlice(Eigen::Index i) const {
        assert(i >= 0 && i < mode &&
               "modeSlice: mode index out of range [0, n).");
        return left_unfolding.middleRows(i * left_rank, left_rank);
    }

    /*!
     * @brief Left-to-right orthogonalization with no incoming R (R = I).
     * @return The upper-triangular factor R (shape r1 x r1) to carry into the
     * next core. The core's unfolding is overwritten by the orthonormal Q.
     */
    Eigen::MatrixX<Scalar> leftOrth() {
        // No incoming R: orthogonalize the left unfolding directly. The bond
        // rank after orthogonalization is q = min(r0 * n, r1); a wide unfolding
        // (r1 > r0 * n) is rank-deficient and the new right rank shrinks to q.
        const Eigen::Index q = std::min(left_rank * mode, right_rank);
        Eigen::ColPivHouseholderQR<Eigen::MatrixX<Scalar>> qr(left_unfolding);

        // Thin Q of shape (r0 * n) x q.
        Eigen::MatrixX<Scalar> Q =
            qr.householderQ() *
            Eigen::MatrixX<Scalar>::Identity(left_rank * mode, q);

        // With column pivoting, A * P = Q * R, so the carry that satisfies
        // A = Q * carry is R * P^T (shape q x r1). The relevant part of the
        // (upper-trapezoidal) R is its first q rows.
        Eigen::MatrixX<Scalar> R =
            qr.matrixR()
                .topLeftCorner(q, right_rank)
                .template triangularView<Eigen::Upper>()
                .toDenseMatrix();
        Eigen::MatrixX<Scalar> carry = R * qr.colsPermutation().transpose();

        // The core is now left-orthogonal: its unfolding becomes Q.
        right_rank = q;
        left_unfolding = std::move(Q);
        return carry;
    }

    /*!
     * @brief Right-to-left orthogonalization with no incoming R (R = I).
     *
     * For the last core of the train, whose right boundary rank is 1 (so there
     * is nothing to absorb). Operates directly on the right unfolding.
     * @return The factor R (shape r0 x r0) to carry into the previous core. The
     * core's unfolding is overwritten by the orthonormal factor.
     */
    Eigen::MatrixX<Scalar> rightOrth() {
        // Right unfolding r0 x (n * r1) viewed directly from the stored buffer.
        Eigen::Map<Eigen::MatrixX<Scalar>> absorbed(
            left_unfolding.data(), left_rank, mode * right_rank);

        // Orthogonalize rows: QR on the transpose (LQ decomposition). The bond
        // rank after orthogonalization is q = min(r0, n * r1); a wide unfolding
        // (r0 > n * r1) is rank-deficient and the new left rank shrinks to q.
        const Eigen::Index q = std::min(left_rank, mode * right_rank);
        Eigen::ColPivHouseholderQR<Eigen::MatrixX<Scalar>> qr(
            absorbed.transpose());

        Eigen::MatrixX<Scalar> Q_t =
            qr.householderQ() *
            Eigen::MatrixX<Scalar>::Identity(mode * right_rank, q);

        Eigen::MatrixX<Scalar> R_qr =
            qr.matrixR()
                .topLeftCorner(q, left_rank)
                .template triangularView<Eigen::Upper>()
                .toDenseMatrix();
        // carry is r0 x q so that old_unfolding = carry * Q_right.
        Eigen::MatrixX<Scalar> carry = qr.colsPermutation() * R_qr.transpose();

        Eigen::MatrixX<Scalar> Q_right = Q_t.transpose(); // q x (n * r1)
        left_rank = q;
        left_unfolding = Eigen::Map<Eigen::MatrixX<Scalar>>(
            Q_right.data(), q * mode, right_rank);
        return carry;
    }

    /*!
     * @brief Left-to-right SVD sweep with rank truncation.
     *
     * Truncation keeps the largest singular values and drops the smallest ones
     * as long as the Frobenius norm of the dropped tail stays within the
     * tolerance (standard TT-SVD rule). The unfolding becomes the orthonormal U.
     * Any incoming factor should be folded in first via `absorbLeftFactor`.
     * @param atol Absolute Frobenius tolerance on the discarded singular
     * values.
     * @param rtol Relative tolerance; the effective threshold is
     * \f$ \max(\text{atol}, \text{rtol} \cdot \lVert \sigma \rVert_2) \f$.
     * @return The carry \f$ \Sigma V^{\top} \f$ (truncated rank x r1) for the
     * next core.
     */
    Eigen::MatrixX<Scalar> leftSvd(Scalar atol, Scalar rtol) {
        Eigen::BDCSVD<Eigen::MatrixX<Scalar>> svd(
            left_unfolding, Eigen::ComputeThinU | Eigen::ComputeThinV);

        Eigen::Index rank = truncatedRank(svd.singularValues(), atol, rtol);

        // U truncated: (r0 * n) x rank.
        Eigen::MatrixX<Scalar> U = svd.matrixU().leftCols(rank);

        // Carry = Sigma * V^T : rank x r1.
        Eigen::MatrixX<Scalar> carry =
            svd.singularValues().head(rank).asDiagonal() *
            svd.matrixV().leftCols(rank).transpose();

        // Update the core: left_unfolding becomes U, right_rank truncated.
        right_rank = rank;
        left_unfolding = std::move(U);
        return carry;
    }

    /*!
     * @brief Right-to-left SVD sweep with rank truncation.
     *
     * The unfolding becomes the orthonormal V^T. Any incoming factor should be
     * folded in first via `absorbRightFactor`.
     * @param atol Absolute Frobenius tolerance on the discarded singular
     * values.
     * @param rtol Relative tolerance; the effective threshold is
     * \f$ \max(\text{atol}, \text{rtol} \cdot \lVert \sigma \rVert_2) \f$.
     * @return The carry \f$ U \Sigma \f$ (r0 x truncated rank) for the previous
     * core.
     */
    Eigen::MatrixX<Scalar> rightSvd(Scalar atol, Scalar rtol) {
        // Right unfolding r0 x (n * r1) viewed directly from the stored buffer.
        Eigen::Map<Eigen::MatrixX<Scalar>> absorbed(
            left_unfolding.data(), left_rank, mode * right_rank);

        Eigen::BDCSVD<Eigen::MatrixX<Scalar>> svd(
            absorbed, Eigen::ComputeThinU | Eigen::ComputeThinV);

        Eigen::Index rank = truncatedRank(svd.singularValues(), atol, rtol);

        // Orthonormal right factor V^T truncated: rank x (n * r1).
        Eigen::MatrixX<Scalar> Vt = svd.matrixV().leftCols(rank).transpose();

        // Carry = U * Sigma : r0 x rank.
        Eigen::MatrixX<Scalar> carry =
            svd.matrixU().leftCols(rank) *
            svd.singularValues().head(rank).asDiagonal();

        // V^T is rank x (n * r1), column-major equal to (rank * n) x r1.
        left_rank = rank;
        left_unfolding = Eigen::Map<Eigen::MatrixX<Scalar>>(
            Vt.data(), rank * mode, right_rank);
        return carry;
    }

    /*!
     * @brief Selects row-indices of the left unfolding for DEIM/cross
     * interpolation.
     * @param R Factor from the previous core's index selection, absorbed into
     * the left-rank dimension to form the candidate unfolding
     * \f$ (R_{\text{rows}} \cdot n) \times r_1 \f$.
     * @param selector Column-subset selector applied to the transpose of the
     * unfolding (columns of the transpose correspond to row-indices of the
     * unfolding).
     * @param oversampling Number of extra indices to select beyond \f$ r_1 \f$.
     * @return `{indices, submatrix}` where `submatrix` is the selected rows of
     * the unfolding (shape selected x r1), to be absorbed as the factor R into
     * the next core's `leftSelectIndices` call.
     */
    std::pair<std::vector<Eigen::Index>, Eigen::MatrixX<Scalar>>
    leftSelectIndices(const Eigen::MatrixX<Scalar> &R,
                      std::unique_ptr<SelectorBase<Scalar>> &selector,
                      Eigen::Index oversampling = 0) const {

        // Transpose so columns correspond to row-indices of the unfolding.
        Eigen::MatrixX<Scalar> X = absorbLeft(R).transpose();
        std::vector<Eigen::Index> indices =
            selector->selectSubset(X, right_rank + oversampling);
        return std::make_pair(indices, X(Eigen::all, indices).transpose());
    }

    /*!
     * @brief Selects column-indices of the right unfolding for DEIM/cross
     * interpolation.
     * @param R Factor from the next core's index selection, absorbed into the
     * right-rank dimension to form the candidate unfolding
     * \f$ r_0 \times (n \cdot R_{\text{cols}}) \f$.
     * @param selector Column-subset selector applied directly to the right
     * unfolding (its columns are the (mode, right-rank) index pairs).
     * @param oversampling Number of extra indices to select beyond \f$ r_0 \f$.
     * @return `{indices, submatrix}` where `submatrix` is the selected columns
     * of the right unfolding (shape r0 x selected), to be absorbed as the
     * factor R into the previous core's `rightSelectIndices` call.
     */
    std::pair<std::vector<Eigen::Index>, Eigen::MatrixX<Scalar>>
    rightSelectIndices(const Eigen::MatrixX<Scalar> &R,
                       std::unique_ptr<SelectorBase<Scalar>> &selector,
                       Eigen::Index oversampling = 0) const {

        // Right unfolding: r0 x (n * R.cols()). Columns are the (mode, r1)
        // index pairs; we select left_rank + oversampling of them.
        Eigen::MatrixX<Scalar> X = absorbRight(R);
        std::vector<Eigen::Index> indices =
            selector->selectSubset(X, left_rank + oversampling);
        return std::make_pair(indices, X(Eigen::all, indices));
    }

    /*!
     * @brief Folds a factor into the right-rank dimension of this core.
     * @param R Factor of shape r1 x r1_new, right-multiplied into the right
     * unfolding.
     *
     * After a left-to-right TT-SVD sweep the last core is left with a trailing
     * carry factor (its right boundary rank is 1) that must be multiplied back
     * in so the train still represents the same tensor. This right-multiplies
     * left_unfolding \f$ (r_0 n) \times r_1 \f$ by \f$ R \f$ and updates
     * `right_rank` to `R.cols()`.
     */
    void absorbRightFactor(const Eigen::MatrixX<Scalar> &R) {
        left_unfolding = left_unfolding * R;
        right_rank = R.cols();
    }

    /*!
     * @brief Folds a factor into the left-rank dimension of this core.
     * @param R Factor of shape r0_new x r0, left-multiplied into the left-rank
     * dimension.
     *
     * Mirror of `absorbRightFactor` for the first core after a right-to-left
     * TT-SVD sweep: absorbs the leading carry factor (its left boundary rank is
     * 1) back into the core. Updates `left_rank` to `R.rows()`.
     */
    void absorbLeftFactor(const Eigen::MatrixX<Scalar> &R) {
        left_unfolding = absorbLeft(R);
        left_rank = R.rows();
    }

  private:
    Eigen::Index left_rank;
    Eigen::Index mode;
    Eigen::Index right_rank;

    // Raw left unfolding, (r0 * n) x r1, column-major.
    Eigen::MatrixX<Scalar> left_unfolding;

    /*!
     * @brief Left-multiplies R into the r0 dimension and returns the resulting
     * left unfolding, shape (R.rows() * n) x r1.
     *
     * With column-major storage, viewing left_unfolding as r0 x (n * r1)
     * exposes the r0 dimension as the leading index; R acts on it directly and
     * the result is re-viewed as the left unfolding.
     */
    Eigen::MatrixX<Scalar> absorbLeft(const Eigen::MatrixX<Scalar> &R) const {
        Eigen::Map<const Eigen::MatrixX<Scalar>> as_r0_leading(
            left_unfolding.data(), left_rank, mode * right_rank);

        Eigen::MatrixX<Scalar> flat = R * as_r0_leading; // R.rows() x (n * r1)

        return Eigen::Map<Eigen::MatrixX<Scalar>>(flat.data(), R.rows() * mode,
                                                  right_rank);
    }

    /*!
     * @brief Right-multiplies R into the r1 dimension and returns the resulting
     * right unfolding, shape r0 x (n * R.cols()).
     */
    Eigen::MatrixX<Scalar> absorbRight(const Eigen::MatrixX<Scalar> &R) const {
        Eigen::Map<const Eigen::MatrixX<Scalar>> right_view(
            left_unfolding.data(), left_rank, mode * right_rank);

        Eigen::MatrixX<Scalar> absorbed(left_rank, mode * R.cols());
        for (Eigen::Index i = 0; i < mode; ++i) {
            absorbed.middleCols(i * R.cols(), R.cols()) =
                right_view.middleCols(i * right_rank, right_rank) * R;
        }
        return absorbed;
    }

    /*!
     * @brief Number of leading singular values to keep under the tail-energy
     * rule.
     *
     * Singular values are non-increasing. Discards the smallest ones while the
     * Frobenius norm of the discarded tail stays below
     * \f$ \max(\text{atol}, \text{rtol} \cdot \lVert \sigma \rVert_2) \f$.
     * At least one singular value is always kept.
     */
    static Eigen::Index truncatedRank(const Eigen::VectorX<Scalar> &sigma,
                                      Scalar atol, Scalar rtol) {
        const Eigen::Index n = sigma.size();
        if (n <= 1) {
            return n;
        }

        const Scalar threshold = std::max(atol, rtol * sigma.norm());
        const Scalar threshold_sq = threshold * threshold;

        // Accumulate the discarded tail energy from the smallest value upward.
        Scalar tail_sq = Scalar(0);
        Eigen::Index rank = n;
        for (Eigen::Index i = n - 1; i >= 1; --i) {
            tail_sq += sigma(i) * sigma(i);
            if (tail_sq > threshold_sq) {
                break;
            }
            rank = i;
        }
        return rank;
    }
};

} // namespace MatSubset::Experiments

#endif // MAT_SUBSET_EXPERIMENTS_TENSOR_TRAIN_CORE_H
