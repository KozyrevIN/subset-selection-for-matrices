#ifndef MAT_SUBSET_EXPERIMENTS_SOLVER_H
#define MAT_SUBSET_EXPERIMENTS_SOLVER_H

#include <cassert>  // For assert
#include <cstddef>  // For std::size_t
#include <deque>    // For std::deque
#include <memory>   // For std::unique_ptr
#include <optional> // For std::optional
#include <utility>  // For std::move
#include <vector>   // For std::vector

#include <Eigen/Core> // For Eigen::Index

#include <MatSubset/MatSubset.h> // For SelectorBase

#include "TTCrossSolver/TensorFibers.h" // For TensorFibers
#include "TTCrossSolver/TensorTrain.h"  // For TensorTrain

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
     */
    [[nodiscard]] virtual TensorFibers<Scalar>
    evaluate(const TensorTrain<Scalar> &state,
             const TensorFibers<Scalar> &state_fibers, Scalar t) const = 0;
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
     */
    [[nodiscard]] virtual TensorFibers<Scalar>
    apply(const TensorFibers<Scalar> &state_fibers, Scalar t) const = 0;
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
 * @brief Explicit time stepper for \f$ \partial_t^p y = F(y, t) \f$ with the
 * state in TT format and every stage evaluated on a shared cross skeleton.
 * @tparam Scalar The underlying scalar type (e.g. `float`, `double`).
 *
 * Each `step()`:
 * 1. Runs `selectIndices` on the current state \f$ y_n \f$ — truncating it at
 *    (`atol`, `rtol`), re-orthogonalizing it, and fixing the step's skeleton
 *    of width (bond rank + `oversampling`).
 * 2. Evaluates the older history states on that skeleton via `atFibers`.
 * 3. Runs the scheme's stage recursion: each stage combines history fibers
 *    and the rhs fibers with the fiber algebra and cross-interpolates the
 *    result back into a (left-orthogonal) train; intermediate stages are
 *    re-sampled on the same skeleton to feed the next rhs evaluation.
 * 4. If a boundary condition is set, it transforms the final stage's fibers
 *    (the new state \f$ y_{n+1} \f$) right before the cross rebuild — the
 *    cheapest possible spot, no extra sampling or rebuild.
 * 5. Rotates the history and advances the time.
 *
 * Rank adaptation: within a step every stage's ranks are capped by the
 * skeleton width, so across steps the ranks can grow by at most
 * `oversampling` per bond per step (and truncation can shrink them);
 * `oversampling = 0` freezes the rank profile.
 */
template <typename Scalar> class Solver {
  public:
    /*!
     * @brief Constructs the solver.
     * @param initial_history The `scheme.history` initial states in
     * chronological order, most recent last (e.g. \f$ \{y_{-1}, y_0\} \f$ for
     * leapfrog, with \f$ y_{-1} \f$ built from the initial conditions).
     * @param rhs The right-hand side \f$ F \f$.
     * @param scheme The time-stepping scheme.
     * @param dt The time step.
     * @param selector Column-subset selector driving the per-step skeleton
     * selection.
     * @param atol Absolute Frobenius tolerance of the per-step TT-SVD
     * truncation.
     * @param rtol Relative tolerance of the per-step truncation.
     * @param oversampling Extra skeleton indices per bond beyond the bond
     * rank; the per-step rank growth budget.
     * @param boundary_condition Optional transform of the new state's fibers
     * at the end of every step (e.g. an absorbing mask); null for none.
     */
    Solver(std::vector<TensorTrain<Scalar>> initial_history,
           std::unique_ptr<RhsBase<Scalar>> rhs, Scheme<Scalar> scheme,
           Scalar dt, std::unique_ptr<SelectorBase<Scalar>> selector,
           Scalar atol, Scalar rtol, Eigen::Index oversampling = 0,
           std::unique_ptr<BoundaryConditionBase<Scalar>> boundary_condition =
               nullptr)
        : rhs(std::move(rhs)), scheme(std::move(scheme)), dt(dt),
          selector(std::move(selector)), atol(atol), rtol(rtol),
          oversampling(oversampling),
          boundary_condition(std::move(boundary_condition)) {
        assert(this->rhs && "Solver: null rhs.");
        assert(this->selector && "Solver: null selector.");
        assert(!this->scheme.stages.empty() &&
               "Solver: the scheme needs at least one stage.");
        assert(this->scheme.history >= 1 &&
               "Solver: the scheme must retain at least the current state.");
        assert((this->scheme.time_order == 1 || this->scheme.time_order == 2) &&
               "Solver: time_order must be 1 or 2.");
        for (const auto &stage : this->scheme.stages) {
            assert(stage.history_weights.size() <= this->scheme.history &&
                   "Solver: a stage references more history than the scheme "
                   "retains.");
        }
        assert(initial_history.size() == this->scheme.history &&
               "Solver: initial history must provide exactly scheme.history "
               "states.");

        // Most recent state last on input, front of the deque internally.
        for (auto it = initial_history.rbegin(); it != initial_history.rend();
             ++it) {
            history.push_back(std::move(*it));
        }
        // selectIndices requires a left-orthogonal train. Only the newest
        // state is ever fed to it: all later states are fibers-constructed
        // and hence left-orthogonal by construction; older initial states are
        // only touched by atFibers, which has no orthogonality requirement.
        history.front().leftOrthogonalize();
    }

    /*! @brief Advances the state by one time step. */
    void step() {
        // Phase 1: refresh the skeleton on y_n. Mutating: truncates at
        // (atol, rtol) and re-orthogonalizes; the returned fibers evaluate
        // the truncated train exactly. The skeleton is fixed for the whole
        // step - every TensorFibers below shares it, as the fiber algebra
        // requires.
        TensorFibers<Scalar> fibers_n =
            history.front().selectIndices(selector, atol, rtol, oversampling);

        // Phase 2: evaluate the older history states on this skeleton. This
        // is what makes multistep schemes work: y_{n-1} was built on the
        // previous step's skeleton, atFibers re-samples it on the current
        // one.
        std::vector<TensorFibers<Scalar>> hist_fibers;
        hist_fibers.reserve(history.size());
        hist_fibers.push_back(fibers_n);
        for (std::size_t m = 1; m < history.size(); ++m) {
            hist_fibers.push_back(history[m].atFibers(fibers_n.skeleton()));
        }

        // dt^p: first-order equations pair the rhs with dt, second-order
        // with dt^2.
        const Scalar dtp = (scheme.time_order == 2) ? dt * dt : dt;

        // Phase 3: stage recursion, Y^(0) = y_n.
        const TensorTrain<Scalar> *stage_state = &history.front();
        TensorFibers<Scalar> stage_fibers = fibers_n;
        std::optional<TensorTrain<Scalar>> next;
        for (std::size_t j = 0; j < scheme.stages.size(); ++j) {
            const auto &stage = scheme.stages[j];

            TensorFibers<Scalar> rhs_fibers = rhs->evaluate(
                *stage_state, stage_fibers, t + stage.rhs_time_offset * dt);
            assert(rhs_fibers.skeleton() == fibers_n.skeleton() &&
                   "Solver: the rhs must return fibers on the state's "
                   "skeleton.");

            // combo = sum_m alpha_m * fibers(y_{n-m}) + gamma * dt^p * F.
            TensorFibers<Scalar> combo = (stage.rhs_weight * dtp) * rhs_fibers;
            for (std::size_t m = 0; m < stage.history_weights.size(); ++m) {
                combo = combo + stage.history_weights[m] * hist_fibers[m];
            }

            // Phase 4: the boundary condition transforms the completed step's
            // fibers (the new state y_{n+1}), not intermediate stages, before
            // the cross rebuild.
            if (j + 1 == scheme.stages.size() && boundary_condition) {
                combo = boundary_condition->apply(combo, t + dt);
                assert(combo.skeleton() == fibers_n.skeleton() &&
                       "Solver: the boundary condition must return fibers on "
                       "the state's skeleton.");
            }

            // Cross-interpolate the fibers back into a train
            // (left-orthogonal by construction).
            next.emplace(combo);
            if (j + 1 < scheme.stages.size()) {
                stage_fibers = next->atFibers(fibers_n.skeleton());
                stage_state = &*next;
            }
        }

        // Phase 5: rotate the history and advance the time.
        history.push_front(std::move(*next));
        while (history.size() > scheme.history) {
            history.pop_back();
        }
        t += dt;
    }

    /*! @brief The current state \f$ y_n \f$ (for saving or comparison). */
    [[nodiscard]] const TensorTrain<Scalar> &getState() const {
        return history.front();
    }

    /*! @brief The current time \f$ t_n \f$ (starts at 0). */
    [[nodiscard]] Scalar time() const { return t; }

  private:
    std::deque<TensorTrain<Scalar>> history; // history[0] = y_n.
    std::unique_ptr<RhsBase<Scalar>> rhs;
    Scheme<Scalar> scheme;
    Scalar dt;
    Scalar t = Scalar(0);
    std::unique_ptr<SelectorBase<Scalar>> selector;
    Scalar atol;
    Scalar rtol;
    Eigen::Index oversampling;
    std::unique_ptr<BoundaryConditionBase<Scalar>> boundary_condition;
};

} // namespace MatSubset::Experiments

#endif // MAT_SUBSET_EXPERIMENTS_SOLVER_H
