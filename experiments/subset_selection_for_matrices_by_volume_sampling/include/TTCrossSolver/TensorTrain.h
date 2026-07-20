#ifndef MAT_SUBSET_EXPERIMENTS_TENSOR_TRAIN_H
#define MAT_SUBSET_EXPERIMENTS_TENSOR_TRAIN_H

#include <cassert>    // For assert
#include <cmath>      // For std::sqrt
#include <cstddef>    // For std::size_t
#include <functional> // For std::function
#include <limits>     // For std::numeric_limits
#include <memory>     // For std::unique_ptr
#include <utility>    // For std::move, std::pair
#include <vector>     // For std::vector

#include <Eigen/Core> // For Eigen::MatrixX, Eigen::Index
#include <Eigen/QR>   // For Eigen::CompleteOrthogonalDecomposition

#include <MatSubset/MatSubset.h> // For SelectorBase

#include "TTCrossSolver/FiberEvaluator.h"  // For FiberEvaluatorBase
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
    // ======================================================================
    //  Construction
    // ======================================================================

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
     * @param atol Absolute Frobenius tolerance of the per-slab rank
     * truncation.
     * @param rtol Relative tolerance of the per-slab rank truncation; the
     * default \f$ \sqrt{\varepsilon} \f$ discards directions with no numerical
     * support in the data.
     *
     * The naive interpolant multiplies each slab by the inverse of the cross
     * matrix \f$ Y(\mathcal{I}^{\le k}, \mathcal{I}^{>k}) \f$, which can be
     * ill-conditioned. Instead, this sweep SVD-factorizes each carry-absorbed
     * slab at (`atol`, `rtol`), keeps the truncated orthonormal factor
     * \f$ Q_k \f$ as the core (so the train is left-orthogonal by
     * construction), discards the rest (it cancels against the cross matrix),
     * and carries the pseudo-inverse of \f$ \hat{Q}_k \f$ into the next slab,
     * where \f$ \hat{Q}_k \f$ collects the rows of the cumulative orthonormal
     * basis \f$ U_{\le k} \f$ at the selected multi-indices, built by the
     * nesting recursion
     * \f$ \hat{Q}_k(j, :) = \hat{Q}_{k-1}(p_j, :) \, Q_k(i_j) \f$
     * over the skeleton's (parent, mode) nodes. These DEIM-selected rows of an
     * orthonormal basis are far better conditioned than the raw cross values.
     * The trailing carry is folded into the last core.
     *
     * The truncation is what keeps an *oversampled* skeleton stable: a slab of
     * numerical rank \f$ r \f$ sampled on a wider skeleton would otherwise
     * hand \f$ \hat{Q}_k \f$ arbitrary null-space directions, whose rows at
     * the skeleton (e.g. the tails of a localized field) can be near zero —
     * the pseudo-inverse then amplifies noise catastrophically. Truncating
     * first turns the rebuild into a well-posed least-squares fit of rank
     * \f$ r \f$.
     */
    explicit TensorTrain(const TensorFibers<Scalar> &fibers,
                         Scalar atol = Scalar(0),
                         Scalar rtol = defaultFiberRtol())
        : TensorTrain(coresFromFibers(fibers, atol, rtol)) {}

    // ======================================================================
    //  Shape getters
    // ======================================================================

    /*! @brief The number of cores (tensor order) \f$ d \f$. */
    [[nodiscard]] std::size_t order() const { return cores.size(); }

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

    // ======================================================================
    //  Core access
    // ======================================================================

    /*! @brief Read-only access to core `k`. */
    [[nodiscard]] const TensorTrainCore<Scalar> &core(std::size_t k) const {
        return cores[k];
    }

    /*!
     * @brief Replaces core `k`.
     * @param k The core index, in `[0, order())`.
     * @param core The replacement core; its ranks must match the neighbours
     * (`core.leftRank() == cores[k-1].rightRank()` and
     * `core.rightRank() == cores[k+1].leftRank()`), and the boundary ranks must
     * stay 1.
     *
     * For testing and experimentation: lets a caller swap in a modified core
     * (e.g. one rebuilt by an accelerated routine) without rebuilding the
     * train.
     */
    void setCore(std::size_t k, TensorTrainCore<Scalar> core) {
        assert(k < cores.size() && "setCore: core index out of range.");
        assert((k > 0 ? core.leftRank() == cores[k - 1].rightRank()
                      : core.leftRank() == 1) &&
               "setCore: left rank must match the left neighbour (1 at the "
               "left boundary).");
        assert((k + 1 < cores.size()
                    ? core.rightRank() == cores[k + 1].leftRank()
                    : core.rightRank() == 1) &&
               "setCore: right rank must match the right neighbour (1 at the "
               "right boundary).");
        cores[k] = std::move(core);
    }

    // ======================================================================
    //  Pointwise evaluation
    // ======================================================================

    /*!
     * @brief Evaluates the train at a single multi-index \f$ (i_1, \dots, i_d)
     * \f$.
     * @param index The mode value for each core, in order; its length must
     * equal the tensor order and each entry must be in range for its core.
     * @return The scalar tensor entry
     * \f$ G_1(i_1) \, G_2(i_2) \cdots G_d(i_d) \f$.
     *
     * Fixing every mode reduces each core to its `modeSlice`, a
     * \f$ r_{k-1} \times r_k \f$ matrix; their product is the \f$ 1 \times 1
     * \f$ entry. Cheap per call, but forms no dense tensor.
     */
    [[nodiscard]] Scalar
    operator()(const std::vector<Eigen::Index> &index) const {
        assert(index.size() == cores.size() &&
               "operator(): index length must equal the tensor order.");
        Eigen::MatrixX<Scalar> acc = cores.front().modeSlice(index.front());
        for (std::size_t k = 1; k < cores.size(); ++k) {
            acc = acc * cores[k].modeSlice(index[k]);
        }
        assert(acc.rows() == 1 && acc.cols() == 1 &&
               "operator(): boundary ranks must be 1.");
        return acc(0, 0);
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
            // mode value i_k = j is the strided set of columns {j, j + n_k,
            // ...}.
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

    // ======================================================================
    //  Structured evaluation
    // ======================================================================

    /*!
     * @brief Evaluates a single slab of the train on a nested cross skeleton
     * (the `FiberEvaluatorBase` contract; see `TrainFiberEvaluator`).
     * @param k The core index, in `[0, order())`.
     * @param skeleton The skeleton to sample on; may be *mixed* — slab `k`
     * only reads left levels `0..k-1` and right levels `k..d-1`, so levels on
     * the other side of an adaptive sweep front are ignored.
     * @return The fiber core \f$ W_{k-1} \, G_k(i_k) \, V_{k+1} \f$ of shape
     * \f$ \ell_{k-1} \times n_k \times \rho_k \f$ — exactly core `k` of
     * `atFibers(skeleton)`.
     *
     * Stateless: each call rebuilds the partial contractions it needs by
     * walking the relevant levels, at \f$ O(d) \f$ small matrix products per
     * call. Use `atFibers` when all slabs of a *fixed* skeleton are wanted —
     * it shares the partials across slabs.
     */
    [[nodiscard]] TensorFibersCore<Scalar>
    atFiber(std::size_t k, const FiberIndices &skeleton) const {
        const std::size_t d = cores.size();
        assert(k < d && "atFiber: core index out of range.");
        assert(skeleton.order() == d &&
               "atFiber: skeleton order must equal the tensor order.");

        // W_{k-1}: cores 0..k-1 at the bond-(k-1) left multi-indices.
        Eigen::MatrixX<Scalar> left_partial =
            Eigen::MatrixX<Scalar>::Identity(1, 1);
        for (std::size_t j = 0; j < k; ++j) {
            left_partial = appendLeftNodes(cores[j], skeleton.leftLevel(j),
                                           left_partial, j == 0);
        }

        // V_{k+1}: cores k+1..d-1 at the bond-k right multi-indices.
        Eigen::MatrixX<Scalar> right_partial =
            Eigen::MatrixX<Scalar>::Identity(1, 1);
        for (std::size_t j = d - 1; j > k; --j) {
            right_partial = prependRightNodes(
                cores[j], skeleton.rightLevel(j - 1), right_partial);
        }

        return formFiberCore(cores[k], left_partial, right_partial);
    }

    /*!
     * @brief Evaluates the train on a fixed nested cross skeleton.
     * @param skeleton The `FiberIndices` to sample on; its levels are read as
     * the (mode, parent) nodes, not re-selected. Its order must equal the
     * train's, and every mode value must be in range for its core.
     * @return `TensorFibers` sharing `skeleton`, whose core `k` holds
     * \f$ W_{k-1} \, G_k(i_k) \, V_{k+1} \f$, a fiber core of shape
     * \f$ \ell_{k-1} \times n_k \times \rho_k \f$ — the train's fibers on the
     * skeleton, exactly the object `selectCross` returns.
     *
     * Read-only counterpart of `selectCross`: it does not select or mutate
     * the cores; the indices are given. Same two passes, but each just
     * contracts the neighbouring cores at the skeleton's nodes rather than
     * sampling them:
     * 1. Right-to-left: builds \f$ V_{k+1} \f$ (`right_partial[k]`, shape
     *    \f$ r_k \times \rho_k \f$), the cores k+1..d-1 evaluated at the bond-k
     *    right multi-indices. Column j of `right_partial[k-1]` prepends mode
     *    \f$ i \f$ to parent p of `right_partial[k]`:
     *    \f$ G_k(i) \, V_{k+1}(:, p) \f$.
     * 2. Left-to-right: builds \f$ W_{k-1} \f$ (`left_partial[k]`, shape
     *    \f$ \ell_{k-1} \times r_{k-1} \f$) symmetrically, then forms the
     * fiber cores.
     */
    [[nodiscard]] TensorFibers<Scalar>
    atFibers(const std::shared_ptr<const FiberIndices> &skeleton) const {
        const std::size_t d = cores.size();
        assert(skeleton && "atFibers: null skeleton.");
        assert(skeleton->order() == d &&
               "atFibers: skeleton order must equal the tensor order.");

        // right_partial[k]: cores k+1..d-1 evaluated at the bond-k right
        // multi-indices (r_k x rho_k); identity at the right boundary.
        std::vector<Eigen::MatrixX<Scalar>> right_partial(d);
        right_partial[d - 1] = Eigen::MatrixX<Scalar>::Identity(1, 1);

        // Pass 1: right-to-left contraction at the skeleton's right nodes.
        for (std::size_t k = d - 1; k >= 1; --k) {
            right_partial[k - 1] = prependRightNodes(
                cores[k], skeleton->rightLevel(k - 1), right_partial[k]);
        }

        // left_partial[k]: cores 0..k-1 evaluated at the bond-(k-1) left
        // multi-indices (l_{k-1} x r_{k-1}); identity at the left boundary.
        std::vector<Eigen::MatrixX<Scalar>> left_partial(d);
        left_partial[0] = Eigen::MatrixX<Scalar>::Identity(1, 1);

        // Pass 2: left-to-right contraction at the skeleton's left nodes.
        for (std::size_t k = 0; k + 1 < d; ++k) {
            left_partial[k + 1] = appendLeftNodes(
                cores[k], skeleton->leftLevel(k), left_partial[k], k == 0);
        }

        // Core k = W_{k-1} * G_k(i) * V_{k+1}, a fiber core of shape l_{k-1} x
        // n_k x rho_k: the train's fibers on the skeleton.
        std::vector<TensorFibersCore<Scalar>> fiber_cores;
        fiber_cores.reserve(d);
        for (std::size_t k = 0; k < d; ++k) {
            fiber_cores.push_back(
                formFiberCore(cores[k], left_partial[k], right_partial[k]));
        }

        return TensorFibers<Scalar>(std::move(fiber_cores), skeleton);
    }

    // ======================================================================
    //  Orthogonalization and compression
    // ======================================================================

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

    // ======================================================================
    //  TT algebra
    // ======================================================================

    /*!
     * @brief Applies this train, read as a TT operator, to a TT tensor by
     * contracting the shared physical mode of every core ("zips" them).
     * @param tensor The TT tensor \f$ B \f$ the operator acts on; must have the
     * same order, and core `k`'s mode size must equal `in_sizes[k]`.
     * @param out_sizes Per-core operator output mode sizes \f$ m_k \f$.
     * @param in_sizes Per-core operator input mode sizes \f$ n_k \f$ (the modes
     * contracted with `tensor`). For each core `out_sizes[k] * in_sizes[k]`
     * must equal this core's mode size, folded row-major as
     * \f$ \text{out} \cdot n_k + \text{in} \f$.
     * @return The result TT of order \f$ d \f$ with mode sizes `out_sizes`; its
     * bond ranks are the products of the two operands' bond ranks (boundary
     * ranks stay 1). Contracting it to dense equals the operator matrix times
     * the tensor's dense vector.
     *
     * Per core this is `TensorTrainCore::zip`; the products of adjacent bond
     * ranks line up because both operands' bonds use the same Kronecker
     * ordering (operator rank outer), so the result is a valid train.
     */
    [[nodiscard]] TensorTrain
    zip(const TensorTrain &tensor, const std::vector<Eigen::Index> &out_sizes,
        const std::vector<Eigen::Index> &in_sizes) const {
        const std::size_t d = cores.size();
        assert(tensor.cores.size() == d &&
               "zip: operands must have the same order.");
        assert(out_sizes.size() == d && in_sizes.size() == d &&
               "zip: out_sizes and in_sizes must have one entry per core.");

        std::vector<TensorTrainCore<Scalar>> out_cores;
        out_cores.reserve(d);
        for (std::size_t k = 0; k < d; ++k) {
            out_cores.push_back(
                cores[k].zip(tensor.cores[k], out_sizes[k], in_sizes[k]));
        }
        return TensorTrain(std::move(out_cores));
    }

    // ======================================================================
    //  Cross interpolation and fiber sampling
    // ======================================================================

    /*!
     * @brief Truncates the train and selects a nested cross skeleton for every
     * bond. Requires no starting set of indices.
     * @param selector Column-subset selector shared across all bonds and both
     * sweeps.
     * @param atol Absolute Frobenius tolerance for the TT-SVD truncation.
     * @param rtol Relative tolerance for the TT-SVD truncation.
     * @param num_samples Skeleton width policy: called per bond as
     * `num_samples(rank, candidates)` with the post-truncation bond rank and
     * the number of candidate indices at that bond, returns how many indices
     * to select. The result is clamped to \f$ [\text{rank},
     * \text{candidates}] \f$. Null selects exactly the bond rank — right for
     * sampling the train itself; give slack (e.g. `ceil(1.1 * rank) + 4`)
     * when the fibers will carry a *combination* whose numerical rank exceeds
     * the train's, such as a time-step update
     * \f$ \sum_m \alpha_m y_{n-m} + \gamma \, \Delta t^p F(y_n) \f$ — a
     * skeleton narrower than that combination's numerical rank makes the
     * rebuild a lossy projection with no tolerance control.
     * @return The selected skeleton.
     *
     * Two sweeps:
     * 1. Right-to-left, mutating: each core absorbs the SVD carry, is
     *    truncated by `rightSvd`, and then its right unfolding — now a
     *    row-orthonormal \f$ V^{\top} \f$, precisely the basis column
     *    selection should run on — is sampled by `rightSelectIndices` with the
     *    partial evaluation \f$ V_{k+1} \f$ of the already-processed cores
     *    absorbed. Selected columns decode into nested (mode, parent) nodes.
     * 2. Left-to-right, mutating: each core absorbs the QR carry and is made
     *    left-orthonormal by `leftOrth` — so its left unfolding is the
     *    orthonormal basis selection should run on — before `leftSelectIndices`
     *    samples its rows with the partial evaluation \f$ W_{k-1} \f$ of the
     *    already-processed cores absorbed. Selected rows decode into nested
     *    (mode, parent) nodes.
     *
     * This is the one routine that manufactures a skeleton from nothing:
     * selection runs on the train's own orthonormalized cores, seeing every
     * candidate index at every bond, whereas the sweeps of `crossInterpolate`
     * only *refine* a skeleton they are given (each slab shows them the
     * tensor through the current skeleton alone). Use it to bootstrap the
     * adaptive sweeps when there is no warm start.
     *
     * @note Assumes the train is left-orthogonal on entry (as produced by the
     * fibers constructor, `leftOrthogonalize()`, or `compress()`); the TT-SVD
     * truncation relies on it. On exit the train is left-orthogonal again, its
     * orthogonality center at the last core.
     */
    std::shared_ptr<const FiberIndices> selectCrossIndices(
        std::unique_ptr<SelectorBase<Scalar>> &selector, Scalar atol,
        Scalar rtol,
        const std::function<Eigen::Index(Eigen::Index, Eigen::Index)>
            &num_samples = nullptr) {
        // Width policy: the bond rank when no policy is given.
        const auto width = [&num_samples](Eigen::Index rank,
                                          Eigen::Index candidates) {
            return num_samples ? num_samples(rank, candidates) : rank;
        };
        const std::size_t d = cores.size();

        std::vector<FiberIndices::Level> left_levels(d);
        std::vector<FiberIndices::Level> right_levels(d);

        // right_partial: cores k+1..d-1 evaluated at the bond-k right
        // multi-indices (r_k x rho_k); identity at the right boundary.
        Eigen::MatrixX<Scalar> right_partial =
            Eigen::MatrixX<Scalar>::Identity(1, 1);
        right_levels[d - 1] = FiberIndices::Level({0}, {-1}); // root node

        // Sweep 1: right-to-left TT-SVD truncation + right index selection.
        Eigen::MatrixX<Scalar> svd_carry;
        for (std::size_t k = d - 1; k >= 1; --k) {
            if (k + 1 < d) {
                cores[k].absorbRightFactor(svd_carry);
            }
            svd_carry = cores[k].rightSvd(atol, rtol);

            const Eigen::Index rho = right_partial.cols();
            auto [indices, submatrix] = cores[k].rightSelectIndices(
                right_partial, selector,
                width(cores[k].leftRank(), cores[k].modeSize() * rho));

            right_levels[k - 1] =
                FiberIndices::Level::fromRightIndices(indices, rho);
            right_partial = std::move(submatrix);
        }
        if (d > 1) {
            cores[0].absorbRightFactor(svd_carry);
        }

        // left_partial: cores 0..k-1 evaluated at the bond-(k-1) left
        // multi-indices (l_{k-1} x r_{k-1}); identity at the left boundary.
        Eigen::MatrixX<Scalar> left_partial =
            Eigen::MatrixX<Scalar>::Identity(1, 1);

        // Sweep 2: left-to-right QR orthogonalization + left index selection.
        // Mirrors sweep 1: each core is made left-orthonormal *before* the
        // selector runs on it, so selection acts on an orthonormal basis (the
        // condition DEIM/volume sampling relies on) rather than on the
        // right-orthogonal cores left by sweep 1. The QR carry is folded into
        // the next core, so the train stays equal to the tensor and ends up
        // left-orthogonal.
        Eigen::MatrixX<Scalar> qr_carry;
        for (std::size_t k = 0; k + 1 < d; ++k) {
            if (k > 0) {
                cores[k].absorbLeftFactor(qr_carry);
            }
            qr_carry = cores[k].leftOrth();

            const Eigen::Index l_prev = left_partial.rows();
            auto [indices, submatrix] = cores[k].leftSelectIndices(
                left_partial, selector,
                width(cores[k].rightRank(), l_prev * cores[k].modeSize()));

            left_levels[k] =
                FiberIndices::Level::fromLeftIndices(indices, l_prev, k == 0);

            left_partial = std::move(submatrix);
        }
        if (d > 1) {
            cores[d - 1].absorbLeftFactor(qr_carry);
        }
        // left_levels[d-1] stays empty: no left set is selected at the last
        // bond (it would enumerate full multi-indices).

        return std::make_shared<const FiberIndices>(std::move(left_levels),
                                                    std::move(right_levels));
    }

    /*!
     * @brief Truncates the train, selects a nested cross skeleton and returns
     * the train's own fibers evaluated on it: `selectCrossIndices` followed by
     * `atFibers`.
     * @param selector,atol,rtol,num_samples See `selectCrossIndices`.
     * @return The train's fibers on the selected skeleton: `TensorFibers` whose
     * fiber core `k` holds \f$ W_{k-1} \, G_k(i_k) \, V_{k+1} \f$, with the
     * `FiberIndices` skeleton embedded (`skeleton()`). Feeding it back into the
     * fibers constructor reproduces the (truncated) train.
     *
     * Mutating exactly like `selectCrossIndices` (truncation +
     * re-orthogonalization); the returned fibers are exact evaluations of the
     * train as it stands on exit.
     */
    TensorFibers<Scalar>
    selectCross(std::unique_ptr<SelectorBase<Scalar>> &selector, Scalar atol,
                Scalar rtol,
                const std::function<Eigen::Index(Eigen::Index, Eigen::Index)>
                    &num_samples = nullptr) {
        return atFibers(selectCrossIndices(selector, atol, rtol, num_samples));
    }

    /*!
     * @brief Builds a train by adaptive TT-cross interpolation of a slab-wise
     * evaluable tensor, refining the skeleton against the tensor itself.
     * @param f The tensor to interpolate, sampled one slab at a time
     * (`FiberEvaluatorBase`).
     * @param skeleton The warm-start skeleton (nested and valid for `f`'s
     * mode sizes); taken by value, refined in place by the sweeps and
     * returned. Its widths need not match `f`'s ranks — each sweep re-selects
     * every level at the width its own truncation prescribes.
     * @param selector Column-subset selector shared by all bonds and sweeps.
     * @param atol Absolute Frobenius tolerance of the per-slab rank
     * truncation.
     * @param rtol Relative tolerance of the per-slab rank truncation.
     * @param num_samples Per-bond skeleton width policy, the same slot as
     * `Solver`/`selectCrossIndices`: called `num_samples(rank, candidates)`
     * with the post-truncation bond rank and the number of candidate fibers,
     * and clamped to \f$ [\text{rank}, \text{candidates}] \f$. Null defaults to
     * `rank + 1`.
     * @param rounds Number of (backward + forward) sweep pairs; 1 suffices
     * for a good warm start (e.g. the previous time step's skeleton), a cold
     * or poor start may need more.
     * @return The interpolant train (left-orthogonal by construction) and the
     * refined skeleton it was sampled on.
     *
     * Each round runs two sweeps over the mutable skeleton:
     * - *Backward*, right-to-left, selection only: slab `k` is evaluated on
     *   the mixed skeleton (stale left levels, right levels refreshed so
     *   far), stabilized by absorbing the carry \f$ \hat{P}_k^+ \f$,
     *   truncated by `rightSvd` — and only then, knowing the truncated rank,
     *   the selector picks `num_samples(rank, candidates)` columns of the
     *   right-orthonormal factor (with \f$ \hat{P}_k \f$ absorbed, so
     *   candidates are actual fibers) as the new bond-(k-1) right level.
     *   \f$ \hat{P} \f$ collects the columns of the cumulative right basis at
     *   the selected fibers, exactly mirroring the \f$ \hat{Q} \f$ recursion
     *   of the fibers constructor.
     * - *Forward*, left-to-right, selection *and* construction: the unified
     *   `forwardSweep` with the dynamic level policy — the index selection
     *   interleaved after each `leftSvd`, so every left level is selected at
     *   the width its core's truncated rank prescribes and the truncated
     *   orthonormal factor is kept as the final core. There is no separate
     *   rebuild that could invalidate the widths afterwards.
     *
     * On exit the left levels are exactly consistent with the returned train;
     * the right levels are one half-sweep stale (their widths reflect the
     * backward sweep's ranks), which the next call's backward sweep corrects
     * first thing — the intended steady state when warm-starting across time
     * steps.
     */
    [[nodiscard]] static std::pair<TensorTrain,
                                   std::shared_ptr<const FiberIndices>>
    crossInterpolate(
        const FiberEvaluatorBase<Scalar> &f, FiberIndices skeleton,
        std::unique_ptr<SelectorBase<Scalar>> &selector, Scalar atol,
        Scalar rtol,
        const std::function<Eigen::Index(Eigen::Index, Eigen::Index)>
            &num_samples = nullptr,
        int rounds = 1) {
        assert(rounds >= 1 && "crossInterpolate: at least one round.");
        assert(skeleton.order() == f.modeSizes().size() &&
               "crossInterpolate: skeleton order must match the tensor "
               "order.");

        // Width policy: rank + 1 when none is given (the historical default
        // oversampling, and the rank-growth headroom per sweep).
        const auto width = [&num_samples](Eigen::Index rank,
                                          Eigen::Index candidates) {
            return num_samples ? num_samples(rank, candidates) : rank + 1;
        };

        std::vector<TensorTrainCore<Scalar>> cores;
        for (int r = 0; r < rounds; ++r) {
            backwardSelectSweep(f, skeleton, selector, atol, rtol, width);
            cores = forwardSweep(
                f, skeleton, atol, rtol,
                [&](std::size_t k, const TensorTrainCore<Scalar> &core,
                    const Eigen::MatrixX<Scalar> &hat_Q) {
                    // Candidate rows are the fibers (p, i): U_{<=k} evaluated
                    // at node p of the bond-(k-1) left level extended by mode
                    // i; width decided after truncation from the truncated
                    // rank and the candidate count.
                    const Eigen::Index l_prev = hat_Q.rows();
                    auto [indices, submatrix] = core.leftSelectIndices(
                        hat_Q, selector,
                        width(core.rightRank(), l_prev * core.modeSize()));
                    skeleton.setLeftLevel(
                        k, FiberIndices::Level::fromLeftIndices(indices, l_prev,
                                                                k == 0));
                    return std::move(submatrix);
                });
        }

        return std::make_pair(
            TensorTrain(std::move(cores)),
            std::make_shared<const FiberIndices>(std::move(skeleton)));
    }

  private:
    std::vector<TensorTrainCore<Scalar>> cores;

    // ----------------------------------------------------------------------
    //  Small utilities
    // ----------------------------------------------------------------------

    /*! @brief Default relative truncation of the fibers rebuild:
     * \f$ \sqrt{\varepsilon} \f$ of the scalar type. */
    static Scalar defaultFiberRtol() {
        return std::sqrt(std::numeric_limits<Scalar>::epsilon());
    }

    /*! @brief Moore-Penrose pseudo-inverse via a complete orthogonal
     * decomposition — the stable inversion of (possibly oversampled) selected
     * rows of an orthonormal basis. */
    static Eigen::MatrixX<Scalar>
    pseudoInverse(const Eigen::MatrixX<Scalar> &A) {
        return Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixX<Scalar>>(A)
            .pseudoInverse();
    }

    // ----------------------------------------------------------------------
    //  Partial-contraction builders (fiber cores and nested fibers)
    // ----------------------------------------------------------------------

    /*!
     * @brief Forms one fiber core \f$ W_{k-1} \, G_k(i) \, V_{k+1} \f$: the
     * train's sampled slab at bond `k`, as a `TensorFibersCore`.
     * @param core The core \f$ G_k \f$.
     * @param left_partial \f$ W_{k-1} \f$, shape \f$ \ell_{k-1} \times r_{k-1}
     * \f$.
     * @param right_partial \f$ V_{k+1} \f$, shape \f$ r_k \times \rho_k \f$.
     * @return The fiber core of shape \f$ \ell_{k-1} \times n_k \times \rho_k
     * \f$, whose left unfolding stacks the mode with the left index fastest.
     */
    static TensorFibersCore<Scalar>
    formFiberCore(const TensorTrainCore<Scalar> &core,
                  const Eigen::MatrixX<Scalar> &left_partial,
                  const Eigen::MatrixX<Scalar> &right_partial) {
        const Eigen::Index l_prev = left_partial.rows();
        const Eigen::Index n = core.modeSize();
        const Eigen::Index rho = right_partial.cols();
        Eigen::MatrixX<Scalar> unfolding(l_prev * n, rho);
        for (Eigen::Index i = 0; i < n; ++i) {
            unfolding.middleRows(i * l_prev, l_prev) =
                left_partial * core.modeSlice(i) * right_partial;
        }
        return TensorFibersCore<Scalar>(std::move(unfolding), n);
    }

    /*!
     * @brief Extends a left partial evaluation by one core, contracted at a
     * left level's (parent, mode) nodes.
     * @param core The core \f$ G_k \f$ the level's mode values index into.
     * @param level The bond-k left level; parents point into the rows of
     * `left_partial`.
     * @param left_partial \f$ W_{k-1} \f$, the previous cores evaluated at the
     * bond-(k-1) left multi-indices (`1 x 1` identity at the left boundary).
     * @param is_root True at bond 0, whose nodes extend the empty boundary
     * index (parent -1) instead of a previous level.
     * @return \f$ W_k \f$ with row `j` \f$ = W_{k-1}(p_j, :) \, G_k(i_j) \f$.
     */
    static Eigen::MatrixX<Scalar>
    appendLeftNodes(const TensorTrainCore<Scalar> &core,
                    const FiberIndices::Level &level,
                    const Eigen::MatrixX<Scalar> &left_partial, bool is_root) {
        Eigen::MatrixX<Scalar> next(static_cast<Eigen::Index>(level.size()),
                                    core.rightRank());
        for (std::size_t j = 0; j < level.size(); ++j) {
            const Eigen::Index i = level.mode(j);
            const Eigen::Index p = is_root ? 0 : level.parentOf(j);
            assert(p >= 0 && p < left_partial.rows() &&
                   "appendLeftNodes: left parent index out of range.");
            next.row(static_cast<Eigen::Index>(j)) =
                left_partial.row(p) * core.modeSlice(i);
        }
        return next;
    }

    /*!
     * @brief Extends a right partial evaluation by one core, contracted at a
     * right level's (mode, parent) nodes; mirror of `appendLeftNodes`.
     * @param core The core \f$ G_k \f$ the level's mode values index into.
     * @param level The bond-(k-1) right level; parents point into the columns
     * of `right_partial`.
     * @param right_partial \f$ V_{k+1} \f$, the later cores evaluated at the
     * bond-k right multi-indices (`1 x 1` identity at the right boundary).
     * @return \f$ V_k \f$ with column `j` \f$ = G_k(i_j) \, V_{k+1}(:, p_j)
     * \f$.
     */
    static Eigen::MatrixX<Scalar>
    prependRightNodes(const TensorTrainCore<Scalar> &core,
                      const FiberIndices::Level &level,
                      const Eigen::MatrixX<Scalar> &right_partial) {
        Eigen::MatrixX<Scalar> next(core.leftRank(),
                                    static_cast<Eigen::Index>(level.size()));
        for (std::size_t j = 0; j < level.size(); ++j) {
            const Eigen::Index i = level.mode(j);
            const Eigen::Index p = level.parentOf(j);
            assert(p >= 0 && p < right_partial.cols() &&
                   "prependRightNodes: right parent index out of range.");
            next.col(static_cast<Eigen::Index>(j)) =
                core.modeSlice(i) * right_partial.col(p);
        }
        return next;
    }

    // ----------------------------------------------------------------------
    //  Cross sweeps (the unified forward engine + the adaptive backward half)
    // ----------------------------------------------------------------------

    /*!
     * @brief The backward half of an adaptive cross round: refreshes every
     * right level of `skeleton` against `f`, right to left.
     *
     * At core `k` the slab is evaluated on the current mixed skeleton,
     * stabilized by absorbing \f$ \hat{P}_k^+ \f$ (the pseudo-inverse of the
     * cumulative right-orthonormal basis at the bond-k right fibers),
     * truncated by `rightSvd`, and the new bond-(k-1) right level is selected
     * from the columns of the resulting right-orthonormal factor with
     * \f$ \hat{P}_k \f$ absorbed — so the candidates are actual fibers
     * (mode, parent into the *new* right level k), keeping the levels nested.
     * The width is `num_samples(rank, candidates)`, decided after the
     * truncation. The cores themselves are discarded; only the skeleton and
     * the \f$ \hat{P} \f$ recursion survive.
     */
    template <typename Width>
    static void
    backwardSelectSweep(const FiberEvaluatorBase<Scalar> &f,
                        FiberIndices &skeleton,
                        std::unique_ptr<SelectorBase<Scalar>> &selector,
                        Scalar atol, Scalar rtol, const Width &width) {
        const std::size_t d = skeleton.order();

        // hat_P: columns of the cumulative right-orthonormal basis V_{>=k+1}
        // at the bond-k right fibers; identity at the right boundary.
        Eigen::MatrixX<Scalar> hat_P = Eigen::MatrixX<Scalar>::Identity(1, 1);
        Eigen::MatrixX<Scalar> carry = Eigen::MatrixX<Scalar>::Identity(1, 1);

        for (std::size_t k = d - 1; k >= 1; --k) {
            TensorFibersCore<Scalar> fiber_core = f.atFiber(k, skeleton);
            assert(fiber_core.leftRank() ==
                       static_cast<Eigen::Index>(skeleton.leftFiberCount(k)) &&
                   "backwardSelectSweep: fiber core left rank must match the "
                   "left fiber count.");

            TensorTrainCore<Scalar> core(fiber_core.leftUnfolding(),
                                         fiber_core.modeSize());
            core.absorbRightFactor(carry); // pinv(hat_P_k), identity at d-1.
            core.rightSvd(atol, rtol);

            // Candidate columns are the fibers (i, p): V_{>=k} evaluated at
            // mode i extending node p of the (new) bond-k right level.
            const Eigen::Index rho = hat_P.cols();
            auto [indices, submatrix] = core.rightSelectIndices(
                hat_P, selector, width(core.leftRank(), core.modeSize() * rho));

            skeleton.setRightLevel(
                k - 1, FiberIndices::Level::fromRightIndices(indices, rho));

            hat_P = std::move(submatrix);
            carry = pseudoInverse(hat_P);
        }
    }

    /*!
     * @brief The unified forward sweep: builds left-orthogonal cores from the
     * slab evaluations of `f` — the stabilized cross rebuild — with the
     * source of each bond's left level as the policy seam.
     * @param f The tensor, sampled one slab at a time on `skeleton`.
     * @param skeleton The skeleton the slabs are evaluated on. Slab `k` reads
     * only left levels `< k` and right levels `>= k`, so a dynamic policy may
     * rewrite left level `k` through its own (non-const) reference at
     * iteration `k` — the mixed-skeleton contract of `FiberEvaluatorBase`.
     * @param next_left_level Invoked at every interior bond `k` — after the
     * slab has absorbed the carry \f$ \hat{Q}_{k-1}^+ \f$ and `leftSvd` has
     * truncated it — with the truncated core and \f$ \hat{Q}_{k-1} \f$.
     * Returns \f$ \hat{Q}_k \f$, the rows of the cumulative orthonormal basis
     * \f$ U_{\le k} \f$ at the bond-k left fibers: by *selecting* a new level
     * into the skeleton (`crossInterpolate`) or by *reading* the skeleton's
     * existing one (the fibers constructor) — the selection submatrix and the
     * `appendLeftNodes` recursion produce the same rows.
     * @return The cores, left-orthogonal by construction (the trailing carry
     * sits in the last core, where no level is produced — a left set at the
     * last bond would enumerate full multi-indices).
     */
    template <typename NextLeftLevel>
    static std::vector<TensorTrainCore<Scalar>>
    forwardSweep(const FiberEvaluatorBase<Scalar> &f,
                 const FiberIndices &skeleton, Scalar atol, Scalar rtol,
                 NextLeftLevel &&next_left_level) {
        const std::size_t d = skeleton.order();
        std::vector<TensorTrainCore<Scalar>> result;
        result.reserve(d);

        // hat_Q: rows of the cumulative orthonormal basis U_{<=k} at the
        // bond-k left fibers; carry = pinv(hat_Q), the stable inversion of
        // DEIM-selected rows of an orthonormal basis (square inverse when not
        // oversampled). At the left boundary both are the 1 x 1 identity.
        Eigen::MatrixX<Scalar> hat_Q = Eigen::MatrixX<Scalar>::Identity(1, 1);
        Eigen::MatrixX<Scalar> carry = Eigen::MatrixX<Scalar>::Identity(1, 1);

        for (std::size_t k = 0; k < d; ++k) {
            TensorFibersCore<Scalar> fiber_core = f.atFiber(k, skeleton);
            assert(fiber_core.leftRank() ==
                       static_cast<Eigen::Index>(skeleton.leftFiberCount(k)) &&
                   "forwardSweep: fiber core left rank must match the left "
                   "fiber count.");

            TensorTrainCore<Scalar> core(fiber_core.leftUnfolding(),
                                         fiber_core.modeSize());
            if (k > 0) {
                assert(carry.cols() == core.leftRank() &&
                       "forwardSweep: carry does not match the fiber core's "
                       "left rank.");
                core.absorbLeftFactor(carry); // pinv(hat_Q_{k-1})
            }
            if (k + 1 == d) {
                // Last core keeps the trailing carry; right boundary rank 1.
                result.push_back(std::move(core));
                break;
            }

            // Keep the truncated orthonormal factor as the core; the carry is
            // never needed (it cancels against the cross matrix in the
            // stabilized formula). Truncating instead of a plain QR is what
            // keeps oversampled skeletons stable: null-space directions of a
            // rank-deficient slab must not reach the pseudo-inverse below.
            core.leftSvd(atol, rtol);

            hat_Q = next_left_level(k, core, hat_Q);
            carry = pseudoInverse(hat_Q);
            result.push_back(std::move(core));
        }
        return result;
    }

    // ----------------------------------------------------------------------
    //  Fibers rebuild
    // ----------------------------------------------------------------------

    /*!
     * @brief Builds left-orthogonal cores from sampled fibers (stabilized
     * TT-cross interpolation): the unified `forwardSweep` with the static
     * level policy, reading the skeleton the fibers were sampled on; see the
     * fibers constructor for the math and the role of the (`atol`, `rtol`)
     * slab truncation.
     */
    static std::vector<TensorTrainCore<Scalar>>
    coresFromFibers(const TensorFibers<Scalar> &fibers, Scalar atol,
                    Scalar rtol) {
        assert(fibers.skeleton() && "TensorTrain(fibers): null skeleton.");
        const FiberIndices &skeleton = *fibers.skeleton();
        const FibersEvaluator<Scalar> evaluator(fibers);

        return forwardSweep(
            evaluator, skeleton, atol, rtol,
            [&skeleton](std::size_t k, const TensorTrainCore<Scalar> &core,
                        const Eigen::MatrixX<Scalar> &hat_Q) {
                // hat_Q_k(j, :) = hat_Q_{k-1}(p_j, :) * Q_k(i_j) over the
                // level's (parent, mode) nodes: the same nesting recursion a
                // dynamic policy gets back as its selection submatrix (the
                // root level's nodes have parent -1, clamped to 0 against the
                // 1 x 1 hat_Q).
                const FiberIndices::Level &level = skeleton.leftLevel(k);
                assert(level.size() > 0 &&
                       "TensorTrain(fibers): empty left index level.");
                return appendLeftNodes(core, level, hat_Q, k == 0);
            });
    }
};

// ----------------------------------------------------------------------
//  FiberEvaluatorBase adapter for tensor trains
// ----------------------------------------------------------------------

/*!
 * @brief Adapts a `TensorTrain` (optionally scaled) to the
 * `FiberEvaluatorBase` interface, for feeding trains into the adaptive cross
 * sweeps and into evaluator combinations.
 * @tparam Scalar The underlying scalar type (e.g. `float`, `double`).
 *
 * Non-owning: the referenced train must outlive the evaluator (evaluators are
 * short-lived sweep inputs by design).
 */
template <typename Scalar>
class TrainFiberEvaluator : public FiberEvaluatorBase<Scalar> {
  public:
    /*!
     * @brief Wraps a train as a slab-wise evaluator of `scale * train`.
     */
    explicit TrainFiberEvaluator(const TensorTrain<Scalar> &train,
                                 Scalar scale = Scalar(1))
        : train(&train), scale(scale) {}

    [[nodiscard]] std::vector<Eigen::Index> modeSizes() const override {
        return train->modeSizes();
    }

    [[nodiscard]] TensorFibersCore<Scalar>
    atFiber(std::size_t k, const FiberIndices &skeleton) const override {
        return scale * train->atFiber(k, skeleton);
    }

  private:
    const TensorTrain<Scalar> *train;
    Scalar scale;
};

// ----------------------------------------------------------------------
//  Tensor trains operators
// ----------------------------------------------------------------------

/*!
 * @brief Sum of two trains by block core concatenation, without truncation.
 *
 * The classic construction: the first cores concatenate horizontally
 * \f$ [A_1(i) \; B_1(i)] \f$, middle cores block-diagonally, the last cores
 * vertically, so every bond rank is the sum of the operands' ranks. Compress
 * afterwards to restore minimal ranks. Per-core work is `TensorTrainCore::add`.
 */
template <typename Scalar>
TensorTrain<Scalar> operator+(const TensorTrain<Scalar> &a,
                              const TensorTrain<Scalar> &b) {
    const std::size_t d = a.order();
    assert(b.order() == d && "TensorTrain operator+: orders must match.");
    assert(a.modeSizes() == b.modeSizes() &&
           "TensorTrain operator+: mode sizes must match.");

    std::vector<TensorTrainCore<Scalar>> cores;
    cores.reserve(d);
    for (std::size_t k = 0; k < d; ++k) {
        cores.push_back(a.core(k).add(b.core(k), k == 0, k + 1 == d));
    }
    return TensorTrain<Scalar>(std::move(cores));
}

/*! @brief Scales a train by a constant (folded into the first core). */
template <typename Scalar>
TensorTrain<Scalar> operator*(Scalar factor, const TensorTrain<Scalar> &a) {
    std::vector<TensorTrainCore<Scalar>> cores;
    cores.reserve(a.order());
    cores.push_back(a.core(0).scale(factor));
    for (std::size_t k = 1; k < a.order(); ++k) {
        cores.push_back(a.core(k));
    }
    return TensorTrain<Scalar>(std::move(cores));
}

/*!
 * @brief Entry-wise (Hadamard) product of two trains, without truncation.
 *
 * Core slices multiply as Kronecker products,
 * \f$ C_k(i) = A_k(i) \otimes B_k(i) \f$, so every bond rank is the product
 * of the operands' ranks. Compress afterwards; a rank-1 factor leaves ranks
 * unchanged. Per-core work is `TensorTrainCore::hadamard`.
 */
template <typename Scalar>
TensorTrain<Scalar> hadamardProduct(const TensorTrain<Scalar> &a,
                                    const TensorTrain<Scalar> &b) {
    const std::size_t d = a.order();
    assert(b.order() == d && "hadamardProduct: orders must match.");
    assert(a.modeSizes() == b.modeSizes() &&
           "hadamardProduct: mode sizes must match.");

    std::vector<TensorTrainCore<Scalar>> cores;
    cores.reserve(d);
    for (std::size_t k = 0; k < d; ++k) {
        cores.push_back(a.core(k).hadamardProduct(b.core(k)));
    }
    return TensorTrain<Scalar>(std::move(cores));
}

} // namespace MatSubset::Experiments

#endif // MAT_SUBSET_EXPERIMENTS_TENSOR_TRAIN_H
