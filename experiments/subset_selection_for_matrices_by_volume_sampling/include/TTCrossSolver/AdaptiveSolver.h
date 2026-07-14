#ifndef MAT_SUBSET_EXPERIMENTS_ADAPTIVE_SOLVER_H
#define MAT_SUBSET_EXPERIMENTS_ADAPTIVE_SOLVER_H

#include <cassert>  // For assert
#include <cstddef>  // For std::size_t
#include <memory>   // For std::unique_ptr, std::shared_ptr
#include <optional> // For std::optional
#include <utility>  // For std::move, std::pair
#include <vector>   // For std::vector

#include <Eigen/Core> // For Eigen::MatrixX, Eigen::Index

#include <MatSubset/MatSubset.h> // For SelectorBase

#include "TTCrossSolver/FiberEvaluator.h" // For FiberEvaluatorBase
#include "TTCrossSolver/SolverBase.h"   // For SolverBase, RhsBase, Scheme
#include "TTCrossSolver/TensorFibers.h"     // For FiberIndices
#include "TTCrossSolver/TensorFibersCore.h" // For TensorFibersCore
#include "TTCrossSolver/TensorTrain.h"      // For TensorTrain

namespace MatSubset::Experiments {

/*!
 * @brief Slab-wise evaluator of one stage combo
 * \f$ \sum_m \alpha_m \, y_{n-m} + \gamma \, \Delta t^p \, F \f$: weighted
 * history trains plus a weighted rhs evaluator, combined entry-wise per slab.
 * @tparam Scalar The underlying scalar type (e.g. `float`, `double`).
 *
 * Non-owning throughout: the history trains and the rhs evaluator must
 * outlive it (it lives for one stage of one step).
 */
template <typename Scalar>
class StageComboEvaluator : public FiberEvaluatorBase<Scalar> {
  public:
    /*!
     * @brief Assembles the combo.
     * @param history_terms `(weight, train)` pairs of the history states
     * entering the stage; zero-weight terms should be omitted by the caller
     * (each term costs a slab evaluation).
     * @param rhs_weight The rhs weight \f$ \gamma \, \Delta t^p \f$.
     * @param rhs_evaluator The stage's rhs evaluator
     * (`RhsBase::makeEvaluator`).
     */
    StageComboEvaluator(
        std::vector<std::pair<Scalar, const TensorTrain<Scalar> *>>
            history_terms,
        Scalar rhs_weight, const FiberEvaluatorBase<Scalar> &rhs_evaluator)
        : history_terms(std::move(history_terms)), rhs_weight(rhs_weight),
          rhs_evaluator(&rhs_evaluator) {}

    [[nodiscard]] std::vector<Eigen::Index> modeSizes() const override {
        return rhs_evaluator->modeSizes();
    }

    [[nodiscard]] TensorFibersCore<Scalar>
    atFiber(std::size_t k, const FiberIndices &skeleton) const override {
        TensorFibersCore<Scalar> combo =
            rhs_weight * rhs_evaluator->atFiber(k, skeleton);
        for (const auto &[weight, train] : history_terms) {
            combo = combo + weight * train->atFiber(k, skeleton);
        }
        return combo;
    }

  private:
    std::vector<std::pair<Scalar, const TensorTrain<Scalar> *>> history_terms;
    Scalar rhs_weight;
    const FiberEvaluatorBase<Scalar> *rhs_evaluator;
};

/*!
 * @brief Explicit time stepper for \f$ \partial_t^p y = F(y, t) \f$ with the
 * state in TT format and every stage rebuilt by *adaptive* TT-cross
 * interpolation of the stage combo itself.
 * @tparam Scalar The underlying scalar type (e.g. `float`, `double`).
 *
 * The adaptive counterpart of `Solver`. Where `Solver` freezes a skeleton
 * from the current state \f$ y_n \f$ and evaluates every stage on it, this
 * solver hands each stage combo — history states plus the rhs, exposed slab
 * by slab through `FiberEvaluatorBase` — to
 * `TensorTrain::crossInterpolate`, which re-selects the skeleton against the
 * combo during its sweeps. The skeleton therefore tracks the structure of
 * what is actually being interpolated (including everything the rhs adds
 * that the state alone does not have), instead of relying on a fixed width
 * slack to cover it.
 *
 * Each stage runs one backward + one forward sweep warm-started from the
 * previous stage's (or step's) skeleton; level widths always equal the
 * post-truncation bond rank plus `oversample` on the side the sweep just
 * refreshed, and one half-sweep stale on the other side — the steady state
 * of warm-started sweeping. `oversample` is also the rank-growth headroom
 * per bond per sweep (a slab is never sampled more than `oversample` wider
 * than the current rank), so 1-2 suffices for slowly-varying dynamics but it
 * stays a knob for resolution studies.
 *
 * The first non-warm-up step bootstraps the skeleton from the current state
 * via `selectIndices` (there is nothing better to warm-start from); warm-up
 * itself lives in `SolverBase` — a structureless initial state gives any
 * selector nothing to work with, adaptive or not.
 */
template <typename Scalar> class AdaptiveSolver : public SolverBase<Scalar> {
  public:
    /*!
     * @brief Constructs the solver.
     * @param initial_history The `scheme.history` initial states in
     * chronological order, most recent last (see `SolverBase`).
     * @param rhs The right-hand side \f$ F \f$; must implement
     * `makeEvaluator` (and `evaluateTrain` if warm-up steps are requested).
     * @param scheme The time-stepping scheme.
     * @param dt The time step.
     * @param selector Column-subset selector driving the index selection of
     * the adaptive sweeps.
     * @param atol Absolute Frobenius tolerance of the per-slab rank
     * truncation.
     * @param rtol Relative tolerance of the per-slab rank truncation.
     * @param oversample Extra indices per level beyond the post-truncation
     * bond rank; also the per-bond rank-growth headroom per sweep.
     * @param boundary_condition Optional transform of the new state at the
     * end of every step; must implement `makeEvaluator` (and `applyTrain`
     * for warm-up). Null for none.
     * @param warmup_steps Number of leading steps taken in exact TT
     * arithmetic (see `SolverBase`).
     */
    AdaptiveSolver(std::vector<TensorTrain<Scalar>> initial_history,
                   std::unique_ptr<RhsBase<Scalar>> rhs, Scheme<Scalar> scheme,
                   Scalar dt, std::unique_ptr<SelectorBase<Scalar>> selector,
                   Scalar atol, Scalar rtol, Eigen::Index oversample = 2,
                   std::unique_ptr<BoundaryConditionBase<Scalar>>
                       boundary_condition = nullptr,
                   int warmup_steps = 0)
        : SolverBase<Scalar>(std::move(initial_history), std::move(rhs),
                             std::move(scheme), dt, std::move(selector), atol,
                             rtol, std::move(boundary_condition), warmup_steps),
          oversample(oversample) {
        assert(oversample >= 0 &&
               "AdaptiveSolver: oversample must be non-negative.");
    }

    /*! @brief The skeleton carried between steps (null until the first
     * adaptive step), for diagnostics. */
    [[nodiscard]] const std::shared_ptr<const FiberIndices> &
    currentSkeleton() const {
        return skeleton;
    }

  protected:
    /*! @brief One fiber-format step: every stage combo cross-interpolated
     * adaptively, the skeleton warm-started across stages and steps. */
    void advance() override {
        // First adaptive step: bootstrap the skeleton from the current state
        // at the widths the adaptive sweeps maintain (rank + oversample).
        // Mutating: truncates y_n at (atol, rtol) and re-orthogonalizes it,
        // exactly like the fixed-skeleton solver's phase 1.
        if (!skeleton) {
            const auto width = [this](Eigen::Index rank, Eigen::Index) {
                return rank + oversample;
            };
            skeleton = history.front()
                           .selectIndices(selector, atol, rtol, width)
                           .skeleton();
        }

        const Scalar dtp = this->dtPower();

        // Stage recursion, Y^(0) = y_n: each stage combo becomes one slab-wise
        // evaluator and is cross-interpolated adaptively, warm-starting the
        // sweeps from the previous stage's (or step's) skeleton.
        const TensorTrain<Scalar> *stage_state = &history.front();
        std::optional<TensorTrain<Scalar>> next;
        for (std::size_t j = 0; j < scheme.stages.size(); ++j) {
            const auto &stage = scheme.stages[j];

            std::unique_ptr<FiberEvaluatorBase<Scalar>> rhs_evaluator =
                rhs->makeEvaluator(*stage_state,
                                   t + stage.rhs_time_offset * dt);
            assert(rhs_evaluator && "AdaptiveSolver: the rhs returned a null "
                                    "evaluator.");

            // combo = sum_m alpha_m * y_{n-m} + gamma * dt^p * F. Zero
            // weights are skipped: each term costs a slab evaluation per
            // sweep visit.
            std::vector<std::pair<Scalar, const TensorTrain<Scalar> *>> terms;
            terms.reserve(stage.history_weights.size());
            for (std::size_t m = 0; m < stage.history_weights.size(); ++m) {
                if (stage.history_weights[m] != Scalar(0)) {
                    terms.emplace_back(stage.history_weights[m], &history[m]);
                }
            }
            StageComboEvaluator<Scalar> combo(
                std::move(terms), stage.rhs_weight * dtp, *rhs_evaluator);

            // The boundary condition wraps the completed step's combo (the
            // new state y_{n+1}), not intermediate stages, so the sweeps
            // select indices for — and interpolate — the transformed state.
            const FiberEvaluatorBase<Scalar> *stage_evaluator = &combo;
            std::unique_ptr<FiberEvaluatorBase<Scalar>> wrapped;
            if (j + 1 == scheme.stages.size() && boundary_condition) {
                wrapped = boundary_condition->makeEvaluator(combo, t + dt);
                assert(wrapped && "AdaptiveSolver: the boundary condition "
                                  "returned a null evaluator.");
                stage_evaluator = &*wrapped;
            }

            auto [train, refined] = TensorTrain<Scalar>::crossInterpolate(
                *stage_evaluator, *skeleton, selector, atol, rtol, oversample);
            skeleton = std::move(refined);
            next.emplace(std::move(train));
            stage_state = &*next;
        }

        this->commitStep(std::move(*next));
    }

  private:
    using SolverBase<Scalar>::history;
    using SolverBase<Scalar>::rhs;
    using SolverBase<Scalar>::scheme;
    using SolverBase<Scalar>::dt;
    using SolverBase<Scalar>::t;
    using SolverBase<Scalar>::selector;
    using SolverBase<Scalar>::atol;
    using SolverBase<Scalar>::rtol;
    using SolverBase<Scalar>::boundary_condition;

    Eigen::Index oversample;

    // The warm-started skeleton, refined by every stage's sweeps; null until
    // the first adaptive step bootstraps it from the state.
    std::shared_ptr<const FiberIndices> skeleton;
};

} // namespace MatSubset::Experiments

#endif // MAT_SUBSET_EXPERIMENTS_ADAPTIVE_SOLVER_H
