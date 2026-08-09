#ifndef MAT_SUBSET_EXPERIMENTS_ALLEN_CAHN_RHS_H
#define MAT_SUBSET_EXPERIMENTS_ALLEN_CAHN_RHS_H

#include <cassert>    // For assert
#include <cstddef>    // For std::size_t
#include <memory>     // For std::unique_ptr, std::make_unique
#include <utility>    // For std::move
#include <vector>     // For std::vector

#include <Eigen/Core> // For Eigen::MatrixX, Eigen::Index

#include <AcousticEquation/AcousticRhs.h>      // For makeOperatorCore
#include <AcousticEquation/FiniteDifference.h> // For centralSecondDerivat...

#include <TTCrossSolver/FiberEvaluator.h>  // For FiberEvaluatorBase
#include <TTCrossSolver/SolverBase.h>      // For RhsBase
#include <TTCrossSolver/TensorFibers.h>    // For TensorFibers
#include <TTCrossSolver/TensorTrain.h>     // For TensorTrain
#include <TTCrossSolver/TensorTrainCore.h> // For TensorTrainCore

namespace MatSubset::Experiments {

/*!
 * @brief The d-dimensional *periodic* second-order finite-difference Laplacian
 * \f$ \Delta = \sum_k I \otimes \dots \otimes L_k \otimes \dots \otimes I \f$
 * as a rank-2 TT operator, for a domain with periodic boundaries.
 * @param sizes The grid points per dimension \f$ (n_1, \dots, n_d) \f$ (the
 * interior nodes of a periodic grid, endpoints excluded — see
 * `AllenCahnDenseSolver` and the driver).
 * @param spacings The grid spacings \f$ (h_1, \dots, h_d) \f$.
 * @param order The (even) accuracy order of the stencil (default 2). Must match
 * `AllenCahnDenseSolver`'s order for the same-grid dense residual to isolate the
 * low-rank error.
 * @return A `TensorTrain` read as a TT operator by `zip` (mode \f$ k \f$ folded
 * as \f$ \text{out} \cdot n_k + \text{in} \f$); interior bond ranks are 2 (1 for
 * d = 1).
 *
 * The twin of `makeLaplacianOperator`, differing only in the 1D stencil: here a
 * term reaching past node \f$ n - 1 \f$ *wraps around* modulo \f$ n \f$ (the
 * periodic boundary) instead of being dropped (the homogeneous Dirichlet
 * boundary). The TT block structure is identical — each axis contributes one
 * \f$ L_k \f$ — so the operator still has bond rank 2 whatever the order. Must
 * match `AllenCahnDenseSolver::laplacian` exactly.
 */
template <typename Scalar>
TensorTrain<Scalar>
makePeriodicLaplacianOperator(const std::vector<Eigen::Index> &sizes,
                              const std::vector<Scalar> &spacings,
                              int order = 2) {
    const std::size_t d = sizes.size();
    assert(d >= 1 && "makePeriodicLaplacianOperator: at least one dimension.");
    assert(spacings.size() == d &&
           "makePeriodicLaplacianOperator: one spacing per dimension.");

    // The 1D central second-derivative stencil of the requested order / h^2,
    // with periodic wrap-around at the ends. Must match
    // AllenCahnDenseSolver::laplacian.
    const std::vector<Scalar> weights =
        centralSecondDerivativeWeights<Scalar>(order);
    const auto stencil = [&weights](Eigen::Index n, Scalar h) {
        Eigen::MatrixX<Scalar> L = Eigen::MatrixX<Scalar>::Zero(n, n);
        const Scalar inv_h2 = Scalar(1) / (h * h);
        const auto radius = static_cast<Eigen::Index>(weights.size()) - 1;
        for (Eigen::Index i = 0; i < n; ++i) {
            L(i, i) += weights[0] * inv_h2;
            for (Eigen::Index j = 1; j <= radius; ++j) {
                const Scalar w = weights[static_cast<std::size_t>(j)] * inv_h2;
                // ((i - j) mod n) and ((i + j) mod n): the periodic wrap. The
                // += (rather than =) matters when the stencil radius reaches
                // more than half way around a tiny axis, so two offsets land
                // on the same node.
                L(i, ((i - j) % n + n) % n) += w;
                L(i, (i + j) % n) += w;
            }
        }
        return L;
    };

    std::vector<TensorTrainCore<Scalar>> cores;
    cores.reserve(d);
    if (d == 1) {
        cores.push_back(
            makeOperatorCore<Scalar>({{stencil(sizes[0], spacings[0])}}));
    } else {
        for (std::size_t k = 0; k < d; ++k) {
            const Eigen::Index n = sizes[k];
            const Eigen::MatrixX<Scalar> L = stencil(n, spacings[k]);
            const Eigen::MatrixX<Scalar> I =
                Eigen::MatrixX<Scalar>::Identity(n, n);
            const Eigen::MatrixX<Scalar> Z = Eigen::MatrixX<Scalar>::Zero(n, n);
            if (k == 0) {
                cores.push_back(makeOperatorCore<Scalar>({{L, I}}));
            } else if (k + 1 < d) {
                cores.push_back(makeOperatorCore<Scalar>({{I, Z}, {L, I}}));
            } else {
                cores.push_back(makeOperatorCore<Scalar>({{I}, {L}}));
            }
        }
    }
    return TensorTrain<Scalar>(std::move(cores));
}

/*!
 * @brief Slab-wise evaluator of the Allen-Cahn rhs
 * \f$ F(f) = \kappa \, \Delta f + f - f^3 \f$ at a fixed stage state, for the
 * adaptive cross sweeps (see `AllenCahnRhs::makeEvaluator`).
 * @tparam Scalar The underlying scalar type (e.g. `float`, `double`).
 *
 * The only term with rank structure, \f$ \Delta f \f$, is formed once at
 * construction by `zip` (it does not depend on the skeleton); the reaction
 * terms \f$ f - f^3 \f$ are the state itself, sampled on the current skeleton
 * and combined entry-wise on the fiber core — the collocation of Dektor
 * (arXiv:2402.18721), one fiber core at a time. The cubic \f$ f^3 \f$ is an
 * exact pointwise Hadamard cube on the skeleton, so no low-rank arithmetic on
 * the nonlinearity is ever performed.
 *
 * Non-owning towards the state: the `TensorTrain` that created it must outlive
 * it (evaluators are consumed within a stage).
 */
template <typename Scalar>
class AllenCahnRhsFiberEvaluator : public FiberEvaluatorBase<Scalar> {
  public:
    /*!
     * @brief Captures the stage: applies the Laplacian to the state and holds
     * the state for the reaction terms.
     * @param laplacian_of_state \f$ \Delta f \f$ as a train (already zipped).
     * @param state The stage state \f$ f \f$; must outlive this evaluator.
     * @param kappa The diffusion coefficient \f$ \kappa \f$.
     */
    AllenCahnRhsFiberEvaluator(TensorTrain<Scalar> laplacian_of_state,
                               const TensorTrain<Scalar> &state, Scalar kappa)
        : laplacian_of_state(std::move(laplacian_of_state)), state(&state),
          kappa(kappa) {}

    [[nodiscard]] std::vector<Eigen::Index> modeSizes() const override {
        return state->modeSizes();
    }

    [[nodiscard]] TensorFibersCore<Scalar>
    atFiber(std::size_t k, const FiberIndices &skeleton) const override {
        // f and kappa * Delta f on the fiber core.
        TensorFibersCore<Scalar> f = state->atFiber(k, skeleton);
        TensorFibersCore<Scalar> diffusion =
            kappa * laplacian_of_state.atFiber(k, skeleton);

        // The reaction f - f^3, entry-wise on the fiber core. The cubic is a
        // pointwise Hadamard cube (Dektor collocation), never a low-rank
        // product. Only one factor of the sum may carry the (-f^3) sign, so we
        // subtract it from the diffusion + f combination below.
        TensorFibersCore<Scalar> f_cubed =
            hadamardProduct(hadamardProduct(f, f), f);

        // kappa * Delta f + f - f^3.
        return diffusion + f + Scalar(-1) * f_cubed;
    }

  private:
    TensorTrain<Scalar> laplacian_of_state;
    const TensorTrain<Scalar> *state;
    Scalar kappa;
};

/*!
 * @brief Right-hand side of the Allen-Cahn equation
 * \f$ \partial_t f = \kappa \, \Delta f + f - f^3 \f$, i.e.
 * \f$ F(f) = \kappa \, \Delta f + f - f^3 \f$, in the fiber format — a
 * first-order-in-time, autonomous, nonlinear reaction-diffusion problem.
 * @tparam Scalar The underlying scalar type (e.g. `float`, `double`).
 *
 * All three terms are evaluated on the step's skeleton (collocation in the
 * sense of Dektor, arXiv:2402.18721), so no low-rank arithmetic on the cubic
 * nonlinearity is ever performed:
 * - \f$ \kappa \, \Delta f \f$: the rank-2 periodic Laplacian TT operator
 *   applied by `zip`, the result sampled with `atFibers` and scaled by
 *   \f$ \kappa \f$ — the only term with rank structure;
 * - \f$ f \f$: the state sampled on the skeleton;
 * - \f$ f^3 \f$: the state sampled on the skeleton, cubed entry-wise via two
 *   `hadamardProduct`s — a pointwise nonlinearity, however large the rank of
 *   \f$ f^3 \f$ would be as a train.
 *
 * There is no time-dependent forcing (the equation is autonomous), so the
 * stage time is ignored throughout.
 */
template <typename Scalar> class AllenCahnRhs : public RhsBase<Scalar> {
  public:
    /*!
     * @brief Constructs the rhs from the diffusion coefficient and the grid.
     * @param kappa The diffusion coefficient \f$ \kappa \f$.
     * @param sizes The grid points per dimension.
     * @param spacings The grid spacings per dimension (for the Laplacian
     * stencil).
     * @param order The (even) accuracy order of the periodic Laplacian stencil
     * (default 2); must match `AllenCahnDenseSolver`'s order for the same-grid
     * dense residual to isolate the low-rank error.
     */
    AllenCahnRhs(Scalar kappa, std::vector<Eigen::Index> sizes,
                 std::vector<Scalar> spacings, int order = 2)
        : kappa(kappa), sizes(std::move(sizes)),
          laplacian(makePeriodicLaplacianOperator<Scalar>(this->sizes, spacings,
                                                          order)) {}

    [[nodiscard]] TensorFibers<Scalar>
    evaluate(const TensorTrain<Scalar> &state,
             const TensorFibers<Scalar> &state_fibers,
             Scalar /*t*/) const override {
        assert(state.modeSizes() == sizes &&
               "AllenCahnRhs: state mode sizes must match the grid.");
        const auto &skeleton = state_fibers.skeleton();

        // kappa * Delta f + f - f^3 on the skeleton: the operator through zip,
        // the reaction terms through the sampled state fibers. The cubic is an
        // exact pointwise Hadamard cube (Dektor collocation).
        TensorFibers<Scalar> diffusion =
            kappa * laplacian.zip(state, sizes, sizes).atFibers(skeleton);
        TensorFibers<Scalar> f_cubed =
            hadamardProduct(hadamardProduct(state_fibers, state_fibers),
                            state_fibers);
        return diffusion + state_fibers + Scalar(-1) * f_cubed;
    }

    [[nodiscard]] TensorTrain<Scalar>
    evaluateTrain(const TensorTrain<Scalar> &state, Scalar /*t*/) const override {
        assert(state.modeSizes() == sizes &&
               "AllenCahnRhs: state mode sizes must match the grid.");
        // The same terms in exact TT arithmetic: the operator inflates ranks by
        // 2, the cube by rank(f)^2, and the caller truncates.
        TensorTrain<Scalar> f_cubed =
            hadamardProduct(hadamardProduct(state, state), state);
        return kappa * laplacian.zip(state, sizes, sizes) + state +
               Scalar(-1) * f_cubed;
    }

    [[nodiscard]] std::unique_ptr<FiberEvaluatorBase<Scalar>>
    makeEvaluator(const TensorTrain<Scalar> &state, Scalar /*t*/) const override {
        assert(state.modeSizes() == sizes &&
               "AllenCahnRhs: state mode sizes must match the grid.");
        // The zip is the skeleton-independent part of the stage; the sweeps
        // then sample its result (and the state) per slab.
        return std::make_unique<AllenCahnRhsFiberEvaluator<Scalar>>(
            laplacian.zip(state, sizes, sizes), state, kappa);
    }

  private:
    Scalar kappa;
    std::vector<Eigen::Index> sizes;
    TensorTrain<Scalar> laplacian;
};

} // namespace MatSubset::Experiments

#endif // MAT_SUBSET_EXPERIMENTS_ALLEN_CAHN_RHS_H
