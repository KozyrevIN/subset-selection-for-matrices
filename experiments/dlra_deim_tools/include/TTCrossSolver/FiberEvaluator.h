#ifndef MAT_SUBSET_EXPERIMENTS_FIBER_EVALUATOR_H
#define MAT_SUBSET_EXPERIMENTS_FIBER_EVALUATOR_H

#include <cstddef> // For std::size_t
#include <vector>  // For std::vector

#include <Eigen/Core> // For Eigen::MatrixX, Eigen::Index

#include "TTCrossSolver/FiberIndices.h"     // For FiberIndices
#include "TTCrossSolver/TensorFibersCore.h" // For TensorFibersCore

namespace MatSubset::Experiments {

/*!
 * @brief Base class for tensors that can be sampled one slab at a time on a
 * cross skeleton — the evaluation interface driven by the adaptive TT-cross
 * sweeps (`TensorTrain::crossInterpolate`).
 * @tparam Scalar The underlying scalar type (e.g. `float`, `double`).
 *
 * `atFiber(k, skeleton)` evaluates slab `k` of the represented tensor: every
 * entry obtained by fixing the left multi-index to a node of the skeleton's
 * bond-(k-1) left level, the right multi-index to a node of its bond-k right
 * level, and letting mode \f$ i_k \f$ run over its full range. Slab `k`
 * therefore depends only on left levels \f$ 0 \dots k-1 \f$ (through the
 * parent chains of level \f$ k-1 \f$) and right levels \f$ k \dots d-1 \f$.
 *
 * That partial dependence is the whole point: the adaptive sweeps hand
 * `atFiber` a *mixed* skeleton, in which the levels behind the sweep front
 * have already been re-selected while the levels ahead still carry the warm
 * start. Implementations must simply evaluate on whatever nested levels the
 * skeleton currently holds; calls arrive in sweep order (k descending for a
 * backward sweep, ascending for a forward sweep), so stateful implementations
 * may cache partial contractions — the provided ones are stateless.
 *
 * Unlike `RhsBase::evaluate`, which consumes and returns whole-skeleton
 * `TensorFibers`, this returns a single fiber core: combining evaluators
 * (sums, pointwise products) do their algebra entry-wise, via the
 * `TensorFibersCore` operators.
 */
template <typename Scalar> class FiberEvaluatorBase {
  public:
    virtual ~FiberEvaluatorBase() = default;

    /*! @brief The mode sizes \f$ (n_1, \dots, n_d) \f$ of the represented
     * tensor. */
    [[nodiscard]] virtual std::vector<Eigen::Index> modeSizes() const = 0;

    /*!
     * @brief Evaluates fiber core `k` on the skeleton as it currently stands.
     * @param k The core index, in `[0, order())`.
     * @param skeleton The (possibly mixed, see the class docs) nested cross
     * skeleton; its order must match `modeSizes().size()` and every relevant
     * mode value must be in range.
     * @return The fiber core of shape
     * `skeleton.leftFiberCount(k) x n_k x skeleton.rightFiberCount(k)`.
     */
    [[nodiscard]] virtual TensorFibersCore<Scalar>
    atFiber(std::size_t k, const FiberIndices &skeleton) const = 0;
};

} // namespace MatSubset::Experiments

#endif // MAT_SUBSET_EXPERIMENTS_FIBER_EVALUATOR_H
