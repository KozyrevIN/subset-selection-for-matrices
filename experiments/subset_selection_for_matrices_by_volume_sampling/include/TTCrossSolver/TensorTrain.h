#ifndef MAT_SUBSET_EXPERIMENTS_TENSOR_TRAIN_H
#define MAT_SUBSET_EXPERIMENTS_TENSOR_TRAIN_H

#include <cassert> // For assert
#include <cstddef> // For std::size_t
#include <memory>  // For std::unique_ptr
#include <utility> // For std::move, std::pair
#include <vector>  // For std::vector

#include <Eigen/Core> // For Eigen::MatrixX, Eigen::Index
#include <Eigen/QR>   // For Eigen::CompleteOrthogonalDecomposition

#include <MatSubset/MatSubset.h> // For SelectorBase

#include "TTCrossSolver/TensorFibers.h"    // For TensorFibers, FiberIndices
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

    /*!
     * @brief Constructs a train from sampled fibers by stabilized TT-cross
     * interpolation, directly in left-orthogonal form.
     * @param fibers The tensor evaluated on a nested cross skeleton (one slab
     * per core plus the `FiberIndices` it was sampled on).
     *
     * The naive interpolant multiplies each slab by the inverse of the cross
     * matrix \f$ Y(\mathcal{I}^{\le k}, \mathcal{I}^{>k}) \f$, which can be
     * ill-conditioned. Instead, this sweep QR-factorizes each carry-absorbed
     * slab, keeps the orthonormal factor \f$ Q_k \f$ as the core (so the train
     * is left-orthogonal by construction, at no extra cost), discards
     * \f$ R_k \f$ (it cancels against the cross matrix), and carries the
     * pseudo-inverse of \f$ \hat{Q}_k \f$ into the next slab, where
     * \f$ \hat{Q}_k \f$ collects the rows of the cumulative orthonormal basis
     * \f$ U_{\le k} \f$ at the selected multi-indices, built by the nesting
     * recursion \f$ \hat{Q}_k(j, :) = \hat{Q}_{k-1}(p_j, :) \, Q_k(i_j) \f$
     * over the skeleton's (parent, mode) nodes. These DEIM-selected rows of an
     * orthonormal basis are far better conditioned than the raw cross values.
     * The trailing carry is folded into the last core.
     */
    explicit TensorTrain(const TensorFibers<Scalar> &fibers)
        : TensorTrain(coresFromFibers(fibers)) {}

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
     * @brief Truncates the train and selects a nested cross skeleton for every
     * bond, returning the train's own fibers evaluated on it.
     * @param selector Column-subset selector shared across all bonds and both
     * sweeps.
     * @param atol Absolute Frobenius tolerance for the TT-SVD truncation.
     * @param rtol Relative tolerance for the TT-SVD truncation.
     * @param oversampling Extra indices selected at each bond beyond the bond
     * rank.
     * @return The train's fibers on the selected skeleton: `TensorFibers` whose
     * slab `k` holds \f$ W_{k-1} \, G_k(i_k) \, V_{k+1} \f$ stacked over the
     * mode, with the `FiberIndices` skeleton embedded (`skeleton()`). Feeding
     * it back into the fibers constructor reproduces the (truncated) train.
     *
     * Two sweeps:
     * 1. Right-to-left, mutating: each core absorbs the SVD carry, is
     *    truncated by `rightSvd`, and then its right unfolding — now a
     *    row-orthonormal \f$ V^{\top} \f$, precisely the basis column
     *    selection should run on — is sampled by `rightSelectIndices` with the
     *    partial evaluation \f$ V_{k+1} \f$ of the already-processed cores
     *    absorbed. Selected columns decode into nested (mode, parent) nodes.
     * 2. Left-to-right, non-mutating: `leftSelectIndices` threads the partial
     *    left evaluations \f$ W_k \f$; selected rows decode likewise.
     *
     * The saved partial evaluations refer to each core after its final
     * mutation, so the returned fibers are exact evaluations of the train as
     * it stands on exit.
     *
     * @note Assumes the train is left-orthogonal on entry (as produced by the
     * fibers constructor, `leftOrthogonalize()`, or `compress()`); the TT-SVD
     * truncation relies on it. On exit the train is right-orthogonal with the
     * orthogonality center at the first core.
     */
    TensorFibers<Scalar>
    selectIndices(std::unique_ptr<SelectorBase<Scalar>> &selector, Scalar atol,
                  Scalar rtol, Eigen::Index oversampling = 0) {
        const std::size_t d = cores.size();

        std::vector<FiberIndices::Level> left_levels(d);
        std::vector<FiberIndices::Level> right_levels(d);

        // right_partial[k]: cores k+1..d-1 evaluated at the bond-k right
        // multi-indices (r_k x rho_k); identity at the right boundary.
        std::vector<Eigen::MatrixX<Scalar>> right_partial(d);
        right_partial[d - 1] = Eigen::MatrixX<Scalar>::Identity(1, 1);
        right_levels[d - 1] = FiberIndices::Level({0}, {-1}); // root node

        // Sweep 1: right-to-left TT-SVD truncation + right index selection.
        Eigen::MatrixX<Scalar> svd_carry;
        for (std::size_t k = d - 1; k >= 1; --k) {
            if (k + 1 < d) {
                cores[k].absorbRightFactor(svd_carry);
            }
            svd_carry = cores[k].rightSvd(atol, rtol);

            const Eigen::Index rho = right_partial[k].cols();
            auto [indices, submatrix] = cores[k].rightSelectIndices(
                right_partial[k], selector, oversampling);

            // Column idx = mode * rho + child in the absorbed right unfolding.
            std::vector<Eigen::Index> modes(indices.size());
            std::vector<Eigen::Index> parents(indices.size());
            for (std::size_t j = 0; j < indices.size(); ++j) {
                modes[j] = indices[j] / rho;
                parents[j] = indices[j] % rho;
            }
            right_levels[k - 1] =
                FiberIndices::Level(std::move(modes), std::move(parents));
            right_partial[k - 1] = std::move(submatrix);
        }
        if (d > 1) {
            cores[0].absorbRightFactor(svd_carry);
        }

        // left_partial[k]: cores 0..k-1 evaluated at the bond-(k-1) left
        // multi-indices (l_{k-1} x r_{k-1}); identity at the left boundary.
        std::vector<Eigen::MatrixX<Scalar>> left_partial(d);
        left_partial[0] = Eigen::MatrixX<Scalar>::Identity(1, 1);

        // Sweep 2: left-to-right index selection (does not mutate the cores).
        for (std::size_t k = 0; k + 1 < d; ++k) {
            const Eigen::Index l_prev = left_partial[k].rows();
            auto [indices, submatrix] = cores[k].leftSelectIndices(
                left_partial[k], selector, oversampling);

            // Row idx = parent + l_prev * mode in the absorbed left unfolding.
            std::vector<Eigen::Index> modes(indices.size());
            std::vector<Eigen::Index> parents(indices.size());
            for (std::size_t j = 0; j < indices.size(); ++j) {
                modes[j] = indices[j] / l_prev;
                // The first level extends the empty boundary index (root -1).
                parents[j] = (k == 0) ? -1 : indices[j] % l_prev;
            }
            left_levels[k] =
                FiberIndices::Level(std::move(modes), std::move(parents));
            left_partial[k + 1] = std::move(submatrix);
        }
        // left_levels[d-1] stays empty: no left set is selected at the last
        // bond (it would enumerate full multi-indices).

        // Slab k = W_{k-1} * G_k(i) * V_{k+1}, stacked over the mode as a left
        // unfolding (l_{k-1} * n_k) x rho_k: the train's fibers.
        std::vector<Eigen::MatrixX<Scalar>> slabs(d);
        for (std::size_t k = 0; k < d; ++k) {
            const Eigen::Index l_prev = left_partial[k].rows();
            const Eigen::Index n = cores[k].modeSize();
            const Eigen::Index rho = right_partial[k].cols();
            slabs[k].resize(l_prev * n, rho);
            for (Eigen::Index i = 0; i < n; ++i) {
                slabs[k].middleRows(i * l_prev, l_prev) =
                    left_partial[k] * cores[k].modeSlice(i) * right_partial[k];
            }
        }

        auto skeleton = std::make_shared<const FiberIndices>(
            std::move(left_levels), std::move(right_levels));
        return TensorFibers<Scalar>(std::move(slabs), std::move(skeleton));
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

    /*!
     * @brief Builds left-orthogonal cores from sampled fibers (stabilized
     * TT-cross interpolation); see the fibers constructor for the math.
     */
    static std::vector<TensorTrainCore<Scalar>>
    coresFromFibers(const TensorFibers<Scalar> &fibers) {
        const FiberIndices &skeleton = *fibers.skeleton();
        const std::size_t d = fibers.order();

        std::vector<TensorTrainCore<Scalar>> result;
        result.reserve(d);

        // hat_Q: rows of the cumulative orthonormal basis U_{<=k} at the
        // selected multi-indices; carry = pinv(hat_Q), folded into the next
        // slab. At the left boundary both are the 1 x 1 identity.
        Eigen::MatrixX<Scalar> hat_Q = Eigen::MatrixX<Scalar>::Identity(1, 1);
        Eigen::MatrixX<Scalar> carry = Eigen::MatrixX<Scalar>::Identity(1, 1);

        for (std::size_t k = 0; k < d; ++k) {
            const auto fiber_count =
                static_cast<Eigen::Index>(skeleton.leftFiberCount(k));
            assert(fibers.slab(k).rows() % fiber_count == 0 &&
                   "TensorTrain(fibers): slab rows must be divisible by the "
                   "left fiber count.");
            const Eigen::Index n = fibers.slab(k).rows() / fiber_count;

            TensorTrainCore<Scalar> core(fibers.slab(k), n);
            if (k > 0) {
                assert(carry.cols() == core.leftRank() &&
                       "TensorTrain(fibers): carry does not match the slab's "
                       "left fiber count.");
                core.absorbLeftFactor(carry);
            }

            if (k + 1 == d) {
                // Last core keeps the trailing carry; right boundary rank 1.
                result.push_back(std::move(core));
                break;
            }

            // Keep the orthonormal factor as the core; R is never needed (it
            // cancels against the cross matrix in the stabilized formula).
            core.leftOrth();

            // hat_Q_k(j, :) = hat_Q_{k-1}(p_j, :) * Q_k(i_j) over the level's
            // (parent, mode) nodes: the selected rows of U_{<=k}.
            const FiberIndices::Level &level = skeleton.leftLevel(k);
            assert(level.size() > 0 &&
                   "TensorTrain(fibers): empty left index level.");
            Eigen::MatrixX<Scalar> next_hat_Q(
                static_cast<Eigen::Index>(level.size()), core.rightRank());
            for (std::size_t j = 0; j < level.size(); ++j) {
                // The root level's nodes have parent -1; hat_Q is 1 x 1 there.
                const Eigen::Index p = (k == 0) ? 0 : level.parentOf(j);
                assert(p >= 0 && p < hat_Q.rows() &&
                       "TensorTrain(fibers): parent index out of range.");
                next_hat_Q.row(static_cast<Eigen::Index>(j)) =
                    hat_Q.row(p) * core.modeSlice(level.mode(j));
            }

            // The stable inversion: pseudo-inverse of DEIM-selected rows of an
            // orthonormal basis (square inverse when not oversampled).
            carry = Eigen::CompleteOrthogonalDecomposition<
                        Eigen::MatrixX<Scalar>>(next_hat_Q)
                        .pseudoInverse();
            hat_Q = std::move(next_hat_Q);
            result.push_back(std::move(core));
        }
        return result;
    }
};

} // namespace MatSubset::Experiments

#endif // MAT_SUBSET_EXPERIMENTS_TENSOR_TRAIN_H
