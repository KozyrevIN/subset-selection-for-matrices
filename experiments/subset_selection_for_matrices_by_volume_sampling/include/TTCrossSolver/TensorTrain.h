#ifndef MAT_SUBSET_EXPERIMENTS_TENSOR_TRAIN_H
#define MAT_SUBSET_EXPERIMENTS_TENSOR_TRAIN_H

#include <cassert> // For assert
#include <cstddef> // For std::size_t
#include <memory>  // For std::unique_ptr
#include <utility> // For std::move, std::pair
#include <vector>  // For std::vector

#include <Eigen/Core> // For Eigen::MatrixX, Eigen::Index

#include <MatSubset/MatSubset.h> // For SelectorBase

#include "TTCrossSolver/TensorTrainCore.h" // For TensorTrainCore

namespace MatSubset::Experiments {

/*!
 * @brief A tensor-train (TT) decomposition: a chain of TT-cores
 * \f$ G_1, \dots, G_d \f$ with \f$ G_k \f$ of shape
 * \f$ r_{k-1} \times n_k \times r_k \f$ and boundary ranks
 * \f$ r_0 = r_d = 1 \f$.
 * @tparam Scalar The underlying scalar type (e.g. `float`, `double`).
 *
 * Every sweep operates on the cores' stored unfoldings directly: a left-to-
 * right QR sweep leaves the train fully left-orthogonal, a right-to-left sweep
 * leaves it fully right-orthogonal, and each core method overwrites its own
 * unfolding with the resulting orthonormal factor.
 */
template <typename Scalar> class TensorTrain {

  public:
    /*!
     * @brief Constructs a train from a list of cores.
     * @param cores The TT-cores, ordered left to right. Adjacent ranks must
     * match (`cores[k].rightRank() == cores[k + 1].leftRank()`) and the
     * boundary ranks must be 1.
     */
    explicit TensorTrain(std::vector<TensorTrainCore<Scalar>> cores)
        : cores(std::move(cores)) {
        assert(!this->cores.empty() &&
               "A tensor-train needs at least one core.");
        assert(this->cores.front().leftRank() == 1 &&
               "The first core must have left boundary rank 1.");
        assert(this->cores.back().rightRank() == 1 &&
               "The last core must have right boundary rank 1.");
        for (std::size_t k = 0; k + 1 < this->cores.size(); ++k) {
            assert(this->cores[k].rightRank() ==
                       this->cores[k + 1].leftRank() &&
                   "Adjacent TT-cores must have matching ranks.");
        }
    }

    /*! @brief The number of cores (tensor order) \f$ d \f$. */
    [[nodiscard]] std::size_t order() const { return cores.size(); }

    /*! @brief Read-only access to core `k`. */
    [[nodiscard]] const TensorTrainCore<Scalar> &core(std::size_t k) const {
        return cores[k];
    }

    /*! @brief The mode sizes \f$ (n_1, \dots, n_d) \f$. */
    [[nodiscard]] std::vector<Eigen::Index> modeSizes() const {
        std::vector<Eigen::Index> sizes(cores.size());
        for (std::size_t k = 0; k < cores.size(); ++k) {
            sizes[k] = cores[k].modeSize();
        }
        return sizes;
    }

    /*! @brief The bond ranks \f$ (r_0, \dots, r_d) \f$, length \f$ d+1 \f$. */
    [[nodiscard]] std::vector<Eigen::Index> ranks() const {
        std::vector<Eigen::Index> r(cores.size() + 1);
        r[0] = cores.front().leftRank();
        for (std::size_t k = 0; k < cores.size(); ++k) {
            r[k + 1] = cores[k].rightRank();
        }
        return r;
    }

    /*!
     * @brief Left-orthogonalizes the train with a left-to-right QR sweep.
     *
     * Every core but the last is made left-orthogonal, its R factor absorbed
     * into the next core. A no-op if the train is already left-orthogonal.
     */
    void leftOrthogonalize() {
        const std::size_t d = cores.size();
        for (std::size_t k = 0; k + 1 < d; ++k) {
            Eigen::MatrixX<Scalar> R = cores[k].leftOrth();
            cores[k + 1].absorbLeftFactor(R);
        }
    }

    /*!
     * @brief Right-orthogonalizes the train with a right-to-left QR sweep.
     *
     * Every core but the first is made right-orthogonal, its R factor absorbed
     * into the previous core. A no-op if the train is already right-orthogonal.
     */
    void rightOrthogonalize() {
        const std::size_t d = cores.size();
        for (std::size_t k = d; k-- > 1;) {
            Eigen::MatrixX<Scalar> R = cores[k].rightOrth();
            cores[k - 1].absorbRightFactor(R);
        }
    }

    /*!
     * @brief Compresses the train with a truncated TT-SVD sweep.
     * @param atol Absolute Frobenius tolerance passed to each core's SVD.
     * @param rtol Relative tolerance passed to each core's SVD.
     *
     * First right-orthogonalizes so the whole norm sits in the first core, then
     * sweeps left to right applying a truncating SVD at every bond. Each SVD
     * carry is folded into the next core before that core is factored; the
     * trailing carry lands in the last core so the tensor is preserved. The
     * train ends up left-orthogonal.
     */
    void compress(Scalar atol, Scalar rtol) {
        rightOrthogonalize();

        const std::size_t d = cores.size();
        Eigen::MatrixX<Scalar> carry = cores.front().leftSvd(atol, rtol);
        for (std::size_t k = 1; k < d; ++k) {
            cores[k].absorbLeftFactor(carry);
            if (k + 1 < d) {
                carry = cores[k].leftSvd(atol, rtol);
            }
        }
    }

    /*!
     * @brief Selects nested cross index sets for every bond of the train.
     * @param selector Column-subset selector shared across all bonds and both
     * sweeps.
     * @param oversampling Extra indices selected at each bond beyond the bond
     * rank.
     * @return A pair `{left_indices, right_indices}`, each of length
     * \f$ d-1 \f$. `left_indices[k]` holds the row-indices chosen on the left
     * unfolding of core `k` (left-to-right nested sets); `right_indices[k]`
     * holds the column-indices chosen on the right unfolding of core `k+1`
     * (right-to-left nested sets).
     *
     * The train is left-orthogonalized first for stability. The sweeps read the
     * raw unfoldings without mutating them, so the orthogonalization state is
     * preserved.
     */
    std::pair<std::vector<std::vector<Eigen::Index>>,
              std::vector<std::vector<Eigen::Index>>>
    selectIndices(std::unique_ptr<SelectorBase<Scalar>> &selector,
                  Eigen::Index oversampling = 0) {
        leftOrthogonalize();

        const std::size_t d = cores.size();
        std::vector<std::vector<Eigen::Index>> left_indices(d - 1);
        std::vector<std::vector<Eigen::Index>> right_indices(d - 1);

        // Left-to-right sweep: nested row-index sets. The first core's left
        // rank is 1, so the incoming factor is the 1 x 1 identity.
        Eigen::MatrixX<Scalar> carry = Eigen::MatrixX<Scalar>::Identity(
            cores.front().leftRank(), cores.front().leftRank());
        for (std::size_t k = 0; k + 1 < d; ++k) {
            auto [indices, submatrix] =
                cores[k].leftSelectIndices(carry, selector, oversampling);
            left_indices[k] = std::move(indices);
            carry = std::move(submatrix);
        }

        // Right-to-left sweep: nested column-index sets. The last core's right
        // rank is 1, so the incoming factor is the 1 x 1 identity.
        carry = Eigen::MatrixX<Scalar>::Identity(cores.back().rightRank(),
                                                 cores.back().rightRank());
        for (std::size_t k = d; k-- > 1;) {
            auto [indices, submatrix] =
                cores[k].rightSelectIndices(carry, selector, oversampling);
            right_indices[k - 1] = std::move(indices);
            carry = std::move(submatrix);
        }

        return {std::move(left_indices), std::move(right_indices)};
    }

    /*!
     * @brief Contracts the train into a full flattened tensor for testing.
     * @return A column vector of length \f$ \prod_k n_k \f$ holding the tensor
     * entries in column-major (first-mode-fastest) order.
     *
     * Not meant for production use; the point of a TT is to avoid forming this.
     */
    [[nodiscard]] Eigen::MatrixX<Scalar> toDense() const {
        // Accumulate the left-to-right contraction as (n_1 ... n_k) x r_k.
        Eigen::MatrixX<Scalar> acc = cores.front().leftUnfolding();
        for (std::size_t k = 1; k < cores.size(); ++k) {
            const Eigen::MatrixX<Scalar> &G = cores[k].leftUnfolding();
            const Eigen::Index rk = cores[k].leftRank(); // = acc.cols()
            const Eigen::Index nk = cores[k].modeSize();
            const Eigen::Index rk1 = cores[k].rightRank();

            // View G as r_{k-1} x (n_k * r_k). Column-major, the column index
            // splits as (i_k, c_out) with i_k fastest, so the slice for a fixed
            // mode value i_k = j is the strided set of columns {j, j + n_k, ...}.
            Eigen::Map<const Eigen::MatrixX<Scalar>> g_right(G.data(), rk,
                                                            nk * rk1);
            // Flatten column-major with i_k slower than the existing modes, to
            // match the tensor entry order i_0 + n_0 i_1 + ... (first-mode
            // fastest).
            Eigen::MatrixX<Scalar> next(acc.rows() * nk, rk1);
            for (Eigen::Index j = 0; j < nk; ++j) {
                Eigen::MatrixX<Scalar> slice(rk, rk1); // G[:, i_k = j, :]
                for (Eigen::Index c = 0; c < rk1; ++c) {
                    slice.col(c) = g_right.col(j + nk * c);
                }
                next.middleRows(j * acc.rows(), acc.rows()) = acc * slice;
            }
            acc = std::move(next);
        }
        return acc; // (n_1 ... n_d) x 1
    }

  private:
    std::vector<TensorTrainCore<Scalar>> cores;
};

} // namespace MatSubset::Experiments

#endif // MAT_SUBSET_EXPERIMENTS_TENSOR_TRAIN_H
