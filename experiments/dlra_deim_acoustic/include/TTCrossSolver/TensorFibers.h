#ifndef MAT_SUBSET_EXPERIMENTS_TENSOR_FIBERS_H
#define MAT_SUBSET_EXPERIMENTS_TENSOR_FIBERS_H

#include <cassert> // For assert
#include <cstddef> // For std::size_t
#include <memory>  // For std::shared_ptr
#include <utility> // For std::move
#include <vector>  // For std::vector

#include <Eigen/Core> // For Eigen::MatrixX, Eigen::Index

#include "TTCrossSolver/FiberEvaluator.h"   // For FiberEvaluatorBase
#include "TTCrossSolver/FiberIndices.h"     // For FiberIndices
#include "TTCrossSolver/TensorFibersCore.h" // For TensorFibersCore

namespace MatSubset::Experiments {

/*!
 * @brief The tensor evaluated on the fibers of a `FiberIndices` skeleton: one
 * slab per core.
 * @tparam Scalar The underlying scalar type (e.g. `float`, `double`).
 *
 * Slab \f$ k \f$ is the 3D tensor of entries of the sampled tensor obtained by
 * fixing the left multi-index to one of `indices->left[k-1]`, the right
 * multi-index to one of `indices->right[k]`, and letting mode \f$ i_k \f$ run
 * over \f$ 0 \dots n_k - 1 \f$. It is stored as a `TensorFibersCore` whose
 * left unfolding has shape \f$ (\ell_{k-1} \, n_k) \times \rho_k \f$, where
 * \f$ \ell_{k-1} = \text{leftFiberCount}(k) \f$ and
 * \f$ \rho_k = \text{rightFiberCount}(k) \f$. These are exactly the shapes
 * consumed by `TensorTrainCore`.
 *
 * Two `TensorFibers` sharing the same `FiberIndices` combine slab-wise via `+`
 * and the Hadamard product, provided as friend free functions delegating to
 * the entry-wise `TensorFibersCore` algebra.
 */
template <typename Scalar> class TensorFibers {
  public:
    TensorFibers() = default;

    /*!
     * @brief Constructs from per-core slab unfoldings and their shared
     * skeleton.
     * @param slabs Slab `k` as a left unfolding, shape
     * `(leftFiberCount(k) * n_k) x rightFiberCount(k)`.
     * @param indices The skeleton the slabs were evaluated on.
     */
    TensorFibers(std::vector<Eigen::MatrixX<Scalar>> slabs,
                 std::shared_ptr<const FiberIndices> indices)
        : indices(std::move(indices)) {
        assert(this->indices && "TensorFibers: null skeleton.");
        assert(slabs.size() == this->indices->order() &&
               "TensorFibers: slab count must equal the tensor order.");
        cores.reserve(slabs.size());
        for (std::size_t k = 0; k < slabs.size(); ++k) {
            const auto l_prev =
                static_cast<Eigen::Index>(this->indices->leftFiberCount(k));
            assert(slabs[k].rows() % l_prev == 0 &&
                   "TensorFibers: slab rows must be divisible by the left "
                   "fiber count.");
            assert(slabs[k].cols() == static_cast<Eigen::Index>(
                                          this->indices->rightFiberCount(k)) &&
                   "TensorFibers: slab columns must equal the right fiber "
                   "count.");
            const Eigen::Index n = slabs[k].rows() / l_prev;
            cores.emplace_back(std::move(slabs[k]), n);
        }
    }

    /*!
     * @brief Constructs from per-core slabs and their shared skeleton.
     * @param cores Slab `k` as a `TensorFibersCore` of shape
     * `leftFiberCount(k) x n_k x rightFiberCount(k)`.
     * @param indices The skeleton the slabs were evaluated on.
     */
    TensorFibers(std::vector<TensorFibersCore<Scalar>> cores,
                 std::shared_ptr<const FiberIndices> indices)
        : cores(std::move(cores)), indices(std::move(indices)) {
        assert(this->indices && "TensorFibers: null skeleton.");
        assert(this->cores.size() == this->indices->order() &&
               "TensorFibers: slab count must equal the tensor order.");
    }

    /*! @brief The tensor order d (number of fiber cores). */
    [[nodiscard]] std::size_t order() const { return cores.size(); }

    /*! @brief Fiber core `k`; its left unfolding (`leftUnfolding()`) is the
     * shape consumed by `TensorTrainCore`. */
    [[nodiscard]] const TensorFibersCore<Scalar> &core(std::size_t k) const {
        assert(k < cores.size() && "TensorFibers: core index OOR.");
        return cores[k];
    }

    /*! @brief The shared skeleton the slabs were sampled on. */
    [[nodiscard]] const std::shared_ptr<const FiberIndices> &skeleton() const {
        return indices;
    }

    /*!
     * @brief Slab-wise sum of two `TensorFibers` sampled on the same skeleton.
     *
     * The pointwise sum of the two sampled fiber tensors, entry by entry on the
     * shared fiber set.
     */
    friend TensorFibers operator+(const TensorFibers &a,
                                  const TensorFibers &b) {
        assertCombinable(a, b);
        std::vector<TensorFibersCore<Scalar>> out;
        out.reserve(a.cores.size());
        for (std::size_t k = 0; k < a.cores.size(); ++k) {
            out.push_back(a.cores[k] + b.cores[k]);
        }
        return TensorFibers(std::move(out), a.indices);
    }

    /*!
     * @brief Scales every slab of a `TensorFibers` by a scalar.
     *
     * The pointwise product of the sampled fiber tensor with `scalar`, entry by
     * entry on the fiber set; the skeleton is shared unchanged.
     */
    friend TensorFibers operator*(Scalar scalar, const TensorFibers &a) {
        std::vector<TensorFibersCore<Scalar>> out;
        out.reserve(a.cores.size());
        for (const auto &core : a.cores) {
            out.push_back(scalar * core);
        }
        return TensorFibers(std::move(out), a.indices);
    }

    /*!
     * @brief Slab-wise elementwise (Hadamard) product of two `TensorFibers`
     * sampled on the same skeleton.
     */
    friend TensorFibers hadamardProduct(const TensorFibers &a,
                                        const TensorFibers &b) {
        assertCombinable(a, b);
        std::vector<TensorFibersCore<Scalar>> out;
        out.reserve(a.cores.size());
        for (std::size_t k = 0; k < a.cores.size(); ++k) {
            out.push_back(hadamardProduct(a.cores[k], b.cores[k]));
        }
        return TensorFibers(std::move(out), a.indices);
    }

  private:
    // Slab k of shape l_{k-1} x n_k x rho_k.
    std::vector<TensorFibersCore<Scalar>> cores;

    // The skeleton the slabs were evaluated on. Shared because a whole family
    // of TensorFibers is typically sampled on one fixed set of indices.
    std::shared_ptr<const FiberIndices> indices;

    /*!
     * @brief Asserts two operands are slab-wise combinable: same order and
     * same shared skeleton (the per-slab shape checks live in the
     * `TensorFibersCore` algebra).
     */
    static void assertCombinable(const TensorFibers &a, const TensorFibers &b) {
        static_cast<void>(a);
        static_cast<void>(b);
        assert(a.cores.size() == b.cores.size() &&
               "TensorFibers combine: mismatched tensor order.");
        assert(a.indices == b.indices &&
               "TensorFibers combine: operands must share the same skeleton.");
    }
};

/*!
 * @brief Adapts already-sampled `TensorFibers` to the `FiberEvaluatorBase`
 * interface: `atFiber` returns the stored slabs instead of evaluating
 * anything.
 * @tparam Scalar The underlying scalar type (e.g. `float`, `double`).
 *
 * Only valid on the exact skeleton the fibers were sampled on, so it feeds
 * the *static*-skeleton sweeps (the fibers constructor of `TensorTrain`) but
 * not the adaptive ones, which re-select levels the stored slabs cannot
 * follow. The counterpart of `TrainFiberEvaluator` for sampled fibers.
 *
 * Non-owning: the referenced fibers must outlive the evaluator (evaluators
 * are short-lived sweep inputs by design).
 */
template <typename Scalar>
class FibersEvaluator : public FiberEvaluatorBase<Scalar> {
  public:
    /*! @brief Wraps sampled fibers as a slab-wise evaluator. */
    explicit FibersEvaluator(const TensorFibers<Scalar> &fibers)
        : fibers(&fibers) {}

    [[nodiscard]] std::vector<Eigen::Index> modeSizes() const override {
        std::vector<Eigen::Index> sizes(fibers->order());
        for (std::size_t k = 0; k < fibers->order(); ++k) {
            sizes[k] = fibers->core(k).modeSize();
        }
        return sizes;
    }

    [[nodiscard]] TensorFibersCore<Scalar>
    atFiber(std::size_t k, const FiberIndices &skeleton) const override {
        assert(&skeleton == fibers->skeleton().get() &&
               "FibersEvaluator: stored slabs are only valid on the skeleton "
               "the fibers were sampled on.");
        static_cast<void>(skeleton);
        return fibers->core(k);
    }

  private:
    const TensorFibers<Scalar> *fibers;
};

} // namespace MatSubset::Experiments

#endif // MAT_SUBSET_EXPERIMENTS_TENSOR_FIBERS_H
