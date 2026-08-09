#ifndef MAT_SUBSET_EXPERIMENTS_SOLVER_BASE_H
#define MAT_SUBSET_EXPERIMENTS_SOLVER_BASE_H

#include <cassert>  // For assert
#include <cstddef>  // For std::size_t
#include <deque>    // For std::deque
#include <memory>   // For std::unique_ptr
#include <optional> // For std::optional
#include <utility>  // For std::move
#include <vector>   // For std::vector

#include <Eigen/Core> // For Eigen::Index

#include <MatSubset/MatSubset.h> // For SelectorBase

#include "TTCrossSolver/FiberEvaluator.h" // For FiberEvaluatorBase
#include "TTCrossSolver/TensorFibers.h"   // For TensorFibers
#include "TTCrossSolver/TensorTrain.h"    // For TensorTrain

namespace MatSubset::Experiments {

/*!
 * @brief Base class for the right-hand side of a time-dependent problem
 * \f$ \partial_t^p y = F(y, t) \f$, evaluated in the fiber format.
 * @tparam Scalar The underlying scalar type (e.g. `float`, `double`).
 */
template <typename Scalar> class RhsBase {
  public:
    virtual ~RhsBase() = default;

    /*!
     * @brief Evaluates \f$ F(y, t) \f$ on the fibers of the state's skeleton.
     * @param state The stage state as a train — for applying TT operators
     * (`zip`) and re-sampling the result (`atFibers`).
     * @param state_fibers The same state evaluated on the step's skeleton —
     * for pointwise terms (`hadamardProduct`) and linear combinations.
     * @param t The stage time.
     * @return The rhs fibers; must share `state_fibers.skeleton()` (build
     * them with `atFibers` on that skeleton and the fiber algebra).
     *
     * Only required by the fixed-skeleton `Solver`; an rhs written for the
     * adaptive solver implements `makeEvaluator` instead. The default
     * asserts.
     */
    [[nodiscard]] virtual TensorFibers<Scalar>
    evaluate(const TensorTrain<Scalar> & /*state*/,
             const TensorFibers<Scalar> &state_fibers, Scalar /*t*/) const {
        assert(false && "RhsBase::evaluate: not implemented by this rhs; "
                        "required by the fixed-skeleton solver.");
        return state_fibers;
    }

    /*!
     * @brief Evaluates \f$ F(y, t) \f$ exactly, as a train, using full TT
     * arithmetic — no skeleton, no sampling.
     * @param state The stage state.
     * @param t The stage time.
     * @return The rhs train (ranks may be much larger than the state's; the
     * caller truncates).
     *
     * Only required when the solver runs warm-up steps; the default asserts.
     * Warm-up exists because the fiber path bootstraps its skeleton from the
     * current state: a structureless initial state (zero, or symmetric noise)
     * gives the selector nothing to work with, so the first steps are taken
     * in exact TT arithmetic until the state carries real structure.
     */
    [[nodiscard]] virtual TensorTrain<Scalar>
    evaluateTrain(const TensorTrain<Scalar> &state, Scalar /*t*/) const {
        assert(false && "RhsBase::evaluateTrain: not implemented by this rhs; "
                        "required for warm-up steps.");
        return state;
    }

    /*!
     * @brief Creates a slab-wise evaluator of \f$ F(\text{state}, t) \f$ for
     * the adaptive cross sweeps (`AdaptiveSolver`,
     * `TensorTrain::crossInterpolate`).
     * @param state The stage state; must outlive the returned evaluator
     * (evaluators are consumed within the stage that creates them).
     * @param t The stage time, fixed at creation.
     * @return An evaluator whose `atFiber(k, skeleton)` samples fiber core `k`
     * of the rhs on whatever (possibly mixed) skeleton the sweep passes in.
     *
     * Called once per stage, so per-stage work that does not depend on the
     * skeleton (e.g. applying TT operators to `state` via `zip`) belongs
     * here, not in `atFiber`. Only required by the adaptive solver; the
     * default asserts.
     */
    [[nodiscard]] virtual std::unique_ptr<FiberEvaluatorBase<Scalar>>
    makeEvaluator(const TensorTrain<Scalar> & /*state*/, Scalar /*t*/) const {
        assert(false && "RhsBase::makeEvaluator: not implemented by this rhs; "
                        "required by the adaptive solver.");
        return nullptr;
    }
};

/*!
 * @brief Base class for boundary conditions applied to the assembled new
 * state at the end of every step, in the fiber format.
 * @tparam Scalar The underlying scalar type (e.g. `float`, `double`).
 *
 * Distinct from the rhs on purpose: anything inside \f$ F \f$ is scaled by
 * \f$ \Delta t^p \f$ and added to the unmodified history, whereas a boundary
 * condition of this kind transforms the *whole* update \f$ y_{n+1} \f$ — e.g.
 * the absorbing sponge \f$ D \odot p^{\text{next}} \f$ of Liu & Sacchi
 * (GeoConvention 2025, Algorithm 1), a pointwise mask that damps the wavefield
 * in a boundary strip after every step.
 */
template <typename Scalar> class BoundaryConditionBase {
  public:
    virtual ~BoundaryConditionBase() = default;

    /*!
     * @brief Transforms the end-of-step state fibers.
     * @param state_fibers The new state \f$ y_{n+1} \f$ evaluated on the
     * step's skeleton, before the cross rebuild.
     * @param t The new time \f$ t_{n+1} \f$.
     * @return The transformed fibers; must share `state_fibers.skeleton()`.
     *
     * Only required by the fixed-skeleton `Solver`; a boundary condition
     * written for the adaptive solver implements `makeEvaluator` instead.
     * The default asserts.
     */
    [[nodiscard]] virtual TensorFibers<Scalar>
    apply(const TensorFibers<Scalar> &state_fibers, Scalar /*t*/) const {
        assert(false &&
               "BoundaryConditionBase::apply: not implemented by this "
               "boundary condition; required by the fixed-skeleton solver.");
        return state_fibers;
    }

    /*!
     * @brief Transforms the end-of-step state exactly, as a train (the
     * warm-up counterpart of `apply`; see `RhsBase::evaluateTrain`).
     *
     * Only required when the solver runs warm-up steps; the default asserts.
     */
    [[nodiscard]] virtual TensorTrain<Scalar>
    applyTrain(const TensorTrain<Scalar> &state, Scalar /*t*/) const {
        assert(false &&
               "BoundaryConditionBase::applyTrain: not implemented by this "
               "boundary condition; required for warm-up steps.");
        return state;
    }

    /*!
     * @brief Wraps a slab-wise evaluator of the new state with this boundary
     * condition, for the adaptive cross sweeps (the slab-wise counterpart of
     * `apply`; see `RhsBase::makeEvaluator`).
     * @param inner The evaluator of the untransformed new state \f$ y_{n+1}
     * \f$; must outlive the returned evaluator.
     * @param t The new time \f$ t_{n+1} \f$, fixed at creation.
     * @return An evaluator whose `atFiber` is the transformed fiber core.
     *
     * Only required by the adaptive solver; the default asserts.
     */
    [[nodiscard]] virtual std::unique_ptr<FiberEvaluatorBase<Scalar>>
    makeEvaluator(const FiberEvaluatorBase<Scalar> & /*inner*/,
                  Scalar /*t*/) const {
        assert(false &&
               "BoundaryConditionBase::makeEvaluator: not implemented by this "
               "boundary condition; required by the adaptive solver.");
        return nullptr;
    }
};

/*!
 * @brief A time-stepping scheme in the unified stage form
 * \f[ Y^{(j)} = \sum_m \alpha_{j,m} \, y_{n-m}
 *     + \gamma_j \, \Delta t^p \, F\big(Y^{(j-1)}, t_n + \theta_j \Delta
 * t\big),
 *     \qquad Y^{(0)} = y_n, \quad y_{n+1} = Y^{(s)}, \f]
 * covering low-storage Runge-Kutta, leapfrog for \f$ \dot y = F \f$ and
 * leapfrog for \f$ \ddot y = F \f$ with one description.
 * @tparam Scalar The underlying scalar type (e.g. `float`, `double`).
 *
 * `time_order` is the order \f$ p \f$ of the time derivative the rhs stands
 * for: stages scale the rhs by \f$ \Delta t \f$ for \f$ \dot y = F \f$ and by
 * \f$ \Delta t^2 \f$ for \f$ \ddot y = F \f$. `history` is the number of past
 * states \f$ y_n, y_{n-1}, \dots \f$ the solver retains across steps.
 *
 * One rhs evaluation per stage, always at the previous stage state — so
 * Adams-Bashforth-style sums over past rhs values are out of scope (past rhs
 * fibers would live on stale skeletons anyway; past *states* re-sample cleanly
 * via `atFibers`, which is why leapfrog fits).
 */
template <typename Scalar> struct Scheme {
    /*! @brief One stage update: history weights, rhs weight, rhs stage time.
     */
    struct Stage {
        /*! @brief \f$ \alpha_{j,m} \f$: weights of \f$ y_{n-m} \f$, most
         * recent first; may be shorter than `history`. */
        std::vector<Scalar> history_weights;
        /*! @brief \f$ \gamma_j \f$: multiplies \f$ \Delta t^p \,
         * F(Y^{(j-1)}) \f$. */
        Scalar rhs_weight;
        /*! @brief \f$ \theta_j \f$: the rhs is evaluated at \f$ t_n + \theta_j
         * \Delta t \f$. */
        Scalar rhs_time_offset;
    };

    std::vector<Stage> stages;
    int time_order = 1;      //!< p: 1 pairs the rhs with dt, 2 with dt^2.
    std::size_t history = 1; //!< Past states retained (1 = only y_n).

    /*! @brief Forward Euler: \f$ y_{n+1} = y_n + \Delta t \, F(y_n, t_n) \f$.
     */
    [[nodiscard]] static Scheme forwardEuler() {
        Scheme scheme;
        scheme.stages = {Stage{{Scalar(1)}, Scalar(1), Scalar(0)}};
        return scheme;
    }

    /*!
     * @brief Low-storage Runge-Kutta with stage coefficients `c`:
     * \f$ Y^{(j)} = y_n + c_j \Delta t \, F(Y^{(j-1)}, t_n + c_{j-1} \Delta t)
     * \f$. `{1}` is forward Euler, `{1/2, 1}` the midpoint rule,
     * `{1/3, 1/2, 1}` third order for linear autonomous problems; the last
     * coefficient should be 1 for consistency.
     */
    [[nodiscard]] static Scheme lowStorageRK(const std::vector<Scalar> &c) {
        assert(!c.empty() && "Scheme: at least one RK coefficient.");
        Scheme scheme;
        scheme.stages.reserve(c.size());
        for (std::size_t j = 0; j < c.size(); ++j) {
            scheme.stages.push_back(
                Stage{{Scalar(1)}, c[j], (j == 0) ? Scalar(0) : c[j - 1]});
        }
        return scheme;
    }

    /*! @brief Leapfrog for \f$ \dot y = F \f$:
     * \f$ y_{n+1} = y_{n-1} + 2 \Delta t \, F(y_n, t_n) \f$. */
    [[nodiscard]] static Scheme leapfrog() {
        Scheme scheme;
        scheme.stages = {Stage{{Scalar(0), Scalar(1)}, Scalar(2), Scalar(0)}};
        scheme.history = 2;
        return scheme;
    }

    /*! @brief Leapfrog for \f$ \ddot y = F \f$:
     * \f$ y_{n+1} = 2 y_n - y_{n-1} + \Delta t^2 \, F(y_n, t_n) \f$. */
    [[nodiscard]] static Scheme leapfrogSecondOrder() {
        Scheme scheme;
        scheme.stages = {Stage{{Scalar(2), Scalar(-1)}, Scalar(1), Scalar(0)}};
        scheme.time_order = 2;
        scheme.history = 2;
        return scheme;
    }
};

/*!
 * @brief Base class of the explicit TT time steppers for
 * \f$ \partial_t^p y = F(y, t) \f$: owns the shared stepping chassis, while
 * children implement how one fiber-format step is taken.
 * @tparam Scalar The underlying scalar type (e.g. `float`, `double`).
 *
 * The chassis comprises the validated scheme, the state history (a deque
 * with \f$ y_n \f$ at the front), the shared tolerances and selector, the
 * time bookkeeping, and the warm-up path — steps taken in exact TT
 * arithmetic (`RhsBase::evaluateTrain`, train algebra, `compress`) while the
 * state is still too structureless for any index selection to work with.
 * `step()` dispatches between warm-up and the child's `advance()`, which
 * performs one step in the fiber format: `Solver` on a skeleton frozen from
 * \f$ y_n \f$, `AdaptiveSolver` by adaptive cross interpolation of each
 * stage combo.
 */
template <typename Scalar> class SolverBase {
  public:
    virtual ~SolverBase() = default;

    SolverBase(const SolverBase &) = delete;
    SolverBase &operator=(const SolverBase &) = delete;

    /*! @brief Advances the state by one time step. */
    void step() {
        if (warmup_remaining > 0) {
            --warmup_remaining;
            warmupStep();
            return;
        }
        advance();
    }

    /*! @brief The current state \f$ y_n \f$ (for saving or comparison). */
    [[nodiscard]] const TensorTrain<Scalar> &getState() const {
        return history.front();
    }

    /*! @brief The current time \f$ t_n \f$ (starts at 0). */
    [[nodiscard]] Scalar time() const { return t; }

  protected:
    /*!
     * @brief Validates and stores the shared stepping state.
     * @param initial_history The `scheme.history` initial states in
     * chronological order, most recent last (e.g. \f$ \{y_{-1}, y_0\} \f$ for
     * leapfrog, with \f$ y_{-1} \f$ built from the initial conditions).
     * @param rhs The right-hand side \f$ F \f$.
     * @param scheme The time-stepping scheme.
     * @param dt The time step.
     * @param selector Column-subset selector driving the index selection.
     * @param atol Absolute Frobenius tolerance of the per-step truncation.
     * @param rtol Relative tolerance of the per-step truncation.
     * @param boundary_condition Optional transform of the new state at the
     * end of every step; null for none.
     * @param warmup_steps Number of leading steps taken in exact TT
     * arithmetic (see the class docs and `RhsBase::evaluateTrain`).
     */
    SolverBase(
        std::vector<TensorTrain<Scalar>> initial_history,
        std::unique_ptr<RhsBase<Scalar>> rhs, Scheme<Scalar> scheme, Scalar dt,
        std::unique_ptr<SelectorBase<Scalar>> selector, Scalar atol,
        Scalar rtol,
        std::unique_ptr<BoundaryConditionBase<Scalar>> boundary_condition,
        int warmup_steps)
        : rhs(std::move(rhs)), scheme(std::move(scheme)), dt(dt),
          selector(std::move(selector)), atol(atol), rtol(rtol),
          boundary_condition(std::move(boundary_condition)),
          warmup_remaining(warmup_steps) {
        assert(warmup_steps >= 0 &&
               "SolverBase: warmup_steps must be non-negative.");
        assert(this->rhs && "SolverBase: null rhs.");
        assert(this->selector && "SolverBase: null selector.");
        assert(!this->scheme.stages.empty() &&
               "SolverBase: the scheme needs at least one stage.");
        assert(this->scheme.history >= 1 &&
               "SolverBase: the scheme must retain at least the current "
               "state.");
        assert((this->scheme.time_order == 1 || this->scheme.time_order == 2) &&
               "SolverBase: time_order must be 1 or 2.");
        for (const auto &stage : this->scheme.stages) {
            static_cast<void>(stage);
            assert(stage.history_weights.size() <= this->scheme.history &&
                   "SolverBase: a stage references more history than the "
                   "scheme retains.");
        }
        assert(initial_history.size() == this->scheme.history &&
               "SolverBase: initial history must provide exactly "
               "scheme.history states.");

        // Most recent state last on input, front of the deque internally.
        for (auto it = initial_history.rbegin(); it != initial_history.rend();
             ++it) {
            history.push_back(std::move(*it));
        }
        // Both fiber paths run selectCross on the newest state (every step
        // for Solver, as the skeleton bootstrap for AdaptiveSolver), which
        // requires a left-orthogonal train. All later states are produced
        // left-orthogonal; older initial states are only touched by fiber
        // evaluation, which has no orthogonality requirement.
        history.front().leftOrthogonalize();
    }

    /*!
     * @brief One step in the fiber format; called by `step()` once warm-up
     * is over. Must end by handing the new state to `commitStep`.
     */
    virtual void advance() = 0;

    /*! @brief \f$ \Delta t^p \f$: first-order equations pair the rhs with
     * dt, second-order with dt^2. */
    [[nodiscard]] Scalar dtPower() const {
        return (scheme.time_order == 2) ? dt * dt : dt;
    }

    /*! @brief Installs the new state \f$ y_{n+1} \f$: rotates the history
     * and advances the time. */
    void commitStep(TensorTrain<Scalar> next) {
        history.push_front(std::move(next));
        while (history.size() > scheme.history) {
            history.pop_back();
        }
        t += dt;
    }

    /*!
     * @brief One step in exact TT arithmetic: the scheme's stage recursion
     * with trains instead of fibers and `compress` instead of a cross
     * rebuild. Exact up to the (`atol`, `rtol`) truncation, whatever the
     * state looks like — at the cost of full TT arithmetic on the
     * (rank-inflated) stage combos.
     */
    void warmupStep() {
        const Scalar dtp = dtPower();

        const TensorTrain<Scalar> *stage_state = &history.front();
        std::optional<TensorTrain<Scalar>> next;
        for (std::size_t j = 0; j < scheme.stages.size(); ++j) {
            const auto &stage = scheme.stages[j];

            TensorTrain<Scalar> combo =
                (stage.rhs_weight * dtp) *
                rhs->evaluateTrain(*stage_state,
                                   t + stage.rhs_time_offset * dt);
            for (std::size_t m = 0; m < stage.history_weights.size(); ++m) {
                // Zero weights are skipped: they contribute nothing but
                // would still inflate the un-truncated combo's ranks.
                if (stage.history_weights[m] != Scalar(0)) {
                    combo = combo + stage.history_weights[m] * history[m];
                }
            }

            if (j + 1 == scheme.stages.size() && boundary_condition) {
                combo = boundary_condition->applyTrain(combo, t + dt);
            }

            // Restore minimal ranks; leaves the train left-orthogonal, as
            // the fiber paths' selectCross requires once warm-up ends.
            combo.compress(atol, rtol);
            next.emplace(std::move(combo));
            stage_state = &*next;
        }

        commitStep(std::move(*next));
    }

    std::deque<TensorTrain<Scalar>> history; // history[0] = y_n.
    std::unique_ptr<RhsBase<Scalar>> rhs;
    Scheme<Scalar> scheme;
    Scalar dt;
    Scalar t = Scalar(0);
    std::unique_ptr<SelectorBase<Scalar>> selector;
    Scalar atol;
    Scalar rtol;
    std::unique_ptr<BoundaryConditionBase<Scalar>> boundary_condition;
    int warmup_remaining;
};

} // namespace MatSubset::Experiments

#endif // MAT_SUBSET_EXPERIMENTS_SOLVER_BASE_H
