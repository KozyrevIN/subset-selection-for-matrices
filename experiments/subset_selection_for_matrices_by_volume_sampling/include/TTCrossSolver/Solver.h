#ifndef MAT_SUBSET_EXPERIMENTS_SOLVER_H
#define MAT_SUBSET_EXPERIMENTS_SOLVER_H

#include <cassert>    // For assert
#include <cstddef>    // For std::size_t
#include <functional> // For std::function
#include <memory>     // For std::unique_ptr
#include <optional>   // For std::optional
#include <utility>    // For std::move
#include <vector>     // For std::vector

#include <Eigen/Core> // For Eigen::Index

#include <MatSubset/MatSubset.h> // For SelectorBase

#include "TTCrossSolver/SolverBase.h"   // For SolverBase, RhsBase, Scheme
#include "TTCrossSolver/TensorFibers.h" // For TensorFibers
#include "TTCrossSolver/TensorTrain.h"  // For TensorTrain

namespace MatSubset::Experiments {

/*!
 * @brief Explicit time stepper for \f$ \partial_t^p y = F(y, t) \f$ with the
 * state in TT format and every stage evaluated on a shared cross skeleton.
 * @tparam Scalar The underlying scalar type (e.g. `float`, `double`).
 *
 * Each fiber-format step (`SolverBase::step` past warm-up):
 * 1. Runs `selectIndices` on the current state \f$ y_n \f$ — truncating it at
 *    (`atol`, `rtol`), re-orthogonalizing it, and fixing the step's skeleton
 *    of the width `num_samples(rank, candidates)` prescribes per bond.
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
 * Skeleton width vs. stage rank: the stage combo
 * \f$ \sum_m \alpha_m y_{n-m} + \gamma \, \Delta t^p F \f$ has numerical rank
 * above the state's — how far above depends on the operators in \f$ F \f$,
 * the history depth, and how many of the new directions exceed `rtol` (which
 * grows with resolution). A skeleton narrower than that rank makes the
 * rebuild a lossy projection whose error is *not* controlled by (`atol`,
 * `rtol`) — it re-enters the state as spurious rank and compounds across
 * steps. `num_samples` sets the width budget; rank-proportional slack (e.g.
 * `ceil(1.1 * rank) + 4`) scales with the problem where a constant offset
 * hides a resolution cliff. The rebuild's per-slab truncation shrinks the
 * state back to its true rank, so a generous width costs fiber evaluations
 * but never inflates the state. (`AdaptiveSolver` removes the guesswork by
 * re-selecting the skeleton against each stage combo itself.)
 */
template <typename Scalar> class Solver : public SolverBase<Scalar> {
  public:
    /*!
     * @brief Constructs the solver.
     * @param initial_history The `scheme.history` initial states in
     * chronological order, most recent last (e.g. \f$ \{y_{-1}, y_0\} \f$ for
     * leapfrog, with \f$ y_{-1} \f$ built from the initial conditions).
     * @param rhs The right-hand side \f$ F \f$; must implement `evaluate`
     * (and `evaluateTrain` if warm-up steps are requested).
     * @param scheme The time-stepping scheme.
     * @param dt The time step.
     * @param selector Column-subset selector driving the per-step skeleton
     * selection.
     * @param atol Absolute Frobenius tolerance of the per-step TT-SVD
     * truncation.
     * @param rtol Relative tolerance of the per-step truncation.
     * @param num_samples Per-bond skeleton width policy
     * `num_samples(rank, candidates)` (see the class docs); null selects
     * exactly the bond rank, freezing the rank profile.
     * @param boundary_condition Optional transform of the new state's fibers
     * at the end of every step (e.g. an absorbing mask); null for none.
     * @param warmup_steps Number of leading steps taken in exact TT
     * arithmetic (see `SolverBase`).
     */
    Solver(std::vector<TensorTrain<Scalar>> initial_history,
           std::unique_ptr<RhsBase<Scalar>> rhs, Scheme<Scalar> scheme,
           Scalar dt, std::unique_ptr<SelectorBase<Scalar>> selector,
           Scalar atol, Scalar rtol,
           std::function<Eigen::Index(Eigen::Index, Eigen::Index)> num_samples =
               nullptr,
           std::unique_ptr<BoundaryConditionBase<Scalar>> boundary_condition =
               nullptr,
           int warmup_steps = 0)
        : SolverBase<Scalar>(std::move(initial_history), std::move(rhs),
                             std::move(scheme), dt, std::move(selector), atol,
                             rtol, std::move(boundary_condition), warmup_steps),
          num_samples(std::move(num_samples)) {}

  protected:
    /*! @brief One fiber-format step on a skeleton frozen from \f$ y_n \f$
     * (phases 1-5 of the class docs). */
    void advance() override {
        // Phase 1: refresh the skeleton on y_n. Mutating: truncates at
        // (atol, rtol) and re-orthogonalizes; the returned fibers evaluate
        // the truncated train exactly. The skeleton is fixed for the whole
        // step - every TensorFibers below shares it, as the fiber algebra
        // requires.
        TensorFibers<Scalar> fibers_n =
            history.front().selectIndices(selector, atol, rtol, num_samples);

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

        const Scalar dtp = this->dtPower();

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
            // (left-orthogonal by construction). The solver's tolerances
            // regularize the rebuild: slab directions below them carry no
            // reliable data and must not survive into the state.
            next.emplace(combo, atol, rtol);
            if (j + 1 < scheme.stages.size()) {
                stage_fibers = next->atFibers(fibers_n.skeleton());
                stage_state = &*next;
            }
        }

        // Phase 5: rotate the history and advance the time.
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

    std::function<Eigen::Index(Eigen::Index, Eigen::Index)> num_samples;
};

} // namespace MatSubset::Experiments

#endif // MAT_SUBSET_EXPERIMENTS_SOLVER_H
