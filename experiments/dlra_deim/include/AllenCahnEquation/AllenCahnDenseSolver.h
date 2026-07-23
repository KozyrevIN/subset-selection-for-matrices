#ifndef MAT_SUBSET_EXPERIMENTS_ALLEN_CAHN_DENSE_SOLVER_H
#define MAT_SUBSET_EXPERIMENTS_ALLEN_CAHN_DENSE_SOLVER_H

#include <cassert>  // For assert
#include <cstddef>  // For std::size_t
#include <deque>    // For std::deque
#include <utility>  // For std::move
#include <vector>   // For std::vector

#include <Eigen/Core> // For Eigen::VectorX, Eigen::Index

#include "AcousticEquation/FiniteDifference.h" // For centralSecondDerivat...
#include "TTCrossSolver/SolverBase.h"          // For Scheme

namespace MatSubset::Experiments {

/*!
 * @brief Full-grid ("dense") reference solver for the Allen-Cahn equation
 * \f$ \partial_t f = \kappa \, \Delta f + f - f^3 \f$ on a periodic domain,
 * stepping the unmodified field with no low-rank compression at all.
 * @tparam Scalar The underlying scalar type (e.g. `float`, `double`).
 *
 * The rank-unlimited ground truth for the TT solvers: it runs the *same*
 * discretization they do — the periodic central second-derivative stencil
 * summed over the axes, the same diffusion coefficient \f$ \kappa \f$, the same
 * pointwise cubic reaction \f$ f - f^3 \f$, and the same time-stepping `Scheme`
 * — but stores the field as a plain flat array and never truncates. Its state
 * is therefore the exact (up to round-off) solution of the discrete problem on
 * this grid, so the difference between it and a TT solver's state is purely
 * that solver's low-rank approximation error.
 *
 * The state layout matches `TensorTrain::toDense()`: a flat vector of length
 * \f$ \prod_k n_k \f$ in first-mode-fastest (column-major) order, so a TT
 * state's `toDense()` and this solver's `field()` are directly comparable
 * node-for-node (see `relativeResidual`).
 *
 * The rhs is baked in — this is not the generic `RhsBase` chassis — because a
 * dense reference only needs to exist for exactly the Allen-Cahn experiment the
 * TT solvers are being checked against. Modelled on `DenseSolver` (the
 * acoustic reference); the differences are the periodic Laplacian wrap and the
 * cubic reaction in place of the wave forcing.
 */
template <typename Scalar> class AllenCahnDenseSolver {
  public:
    /*!
     * @brief Constructs the solver from the diffusion coefficient and the grid.
     * @param kappa The diffusion coefficient \f$ \kappa \f$.
     * @param sizes The grid points per dimension \f$ (n_1, \dots, n_d) \f$.
     * @param spacings The grid spacings \f$ (h_1, \dots, h_d) \f$ (for the
     * Laplacian stencil).
     * @param scheme The time-stepping scheme (e.g. `Scheme::lowStorageRK`).
     * @param dt The time step.
     * @param initial_history The `scheme.history` initial states as flat
     * fields, in chronological order, most recent last; each of length
     * \f$ \prod_k n_k \f$. Allen-Cahn is first order in time, so this is
     * typically the single initial field \f$ \{f_0\} \f$.
     * @param order The (even) accuracy order of the Laplacian stencil
     * (default 2); must match the `AllenCahnRhs` order of the TT solvers being
     * compared for the same-grid dense residual to isolate the low-rank error.
     */
    AllenCahnDenseSolver(Scalar kappa, std::vector<Eigen::Index> sizes,
                         std::vector<Scalar> spacings, Scheme<Scalar> scheme,
                         Scalar dt,
                         std::vector<Eigen::VectorX<Scalar>> initial_history,
                         int order = 2)
        : kappa(kappa), sizes(std::move(sizes)), spacings(std::move(spacings)),
          scheme(std::move(scheme)), dt(dt),
          stencil_weights(centralSecondDerivativeWeights<Scalar>(order)) {
        assert(!this->scheme.stages.empty() &&
               "AllenCahnDenseSolver: the scheme needs at least one stage.");
        assert(this->scheme.time_order == 1 &&
               "AllenCahnDenseSolver: Allen-Cahn is first order in time.");
        assert(this->sizes.size() == this->spacings.size() &&
               "AllenCahnDenseSolver: one spacing per dimension.");

        total_size = 1;
        for (const Eigen::Index n : this->sizes) {
            assert(n >= 1 && "AllenCahnDenseSolver: each axis needs >= 1 point.");
            total_size *= n;
        }
        assert(initial_history.size() == this->scheme.history &&
               "AllenCahnDenseSolver: initial history must provide exactly "
               "scheme.history states.");

        // Most recent state last on input, front of the deque internally
        // (history[0] = f_n), matching SolverBase's convention.
        for (auto it = initial_history.rbegin(); it != initial_history.rend();
             ++it) {
            assert(it->size() == total_size &&
                   "AllenCahnDenseSolver: each initial state size must equal "
                   "prod(sizes).");
            history.push_back(std::move(*it));
        }
    }

    /*! @brief Advances the field by one time step. */
    void step() {
        const Scalar dtp = dt; // First order in time: rhs pairs with dt.

        // Stage recursion Y^(0) = f_n, exactly as SolverBase but on flat fields
        // with no truncation.
        Eigen::VectorX<Scalar> stage_state = history.front();
        for (std::size_t j = 0; j < scheme.stages.size(); ++j) {
            const auto &stage = scheme.stages[j];

            // combo = sum_m alpha_m f_{n-m} + gamma dt F(Y^(j-1)).
            Eigen::VectorX<Scalar> combo =
                (stage.rhs_weight * dtp) * rhs(stage_state);
            for (std::size_t m = 0; m < stage.history_weights.size(); ++m) {
                if (stage.history_weights[m] != Scalar(0)) {
                    combo += stage.history_weights[m] * history[m];
                }
            }
            stage_state = std::move(combo);
        }

        history.push_front(std::move(stage_state));
        while (history.size() > scheme.history) {
            history.pop_back();
        }
        t += dt;
    }

    /*! @brief The current field \f$ f_n \f$ as a flat vector, first-mode-fastest
     * (comparable node-for-node with `TensorTrain::toDense()`). */
    [[nodiscard]] const Eigen::VectorX<Scalar> &field() const {
        return history.front();
    }

    /*! @brief The current time \f$ t_n \f$ (starts at 0). */
    [[nodiscard]] Scalar time() const { return t; }

    /*!
     * @brief The relative \f$ L^2 \f$ residual of a candidate field against
     * this (reference) field: \f$ \lVert f - f_{\text{ref}} \rVert_2 /
     * \lVert f_{\text{ref}} \rVert_2 \f$.
     * @param candidate A flat field in the same layout and of the same length
     * as `field()` (e.g. a TT state's `toDense()`).
     * @return The relative error; falls back to the absolute norm of the
     * difference when the reference field is (numerically) zero.
     */
    [[nodiscard]] Scalar
    relativeResidual(const Eigen::VectorX<Scalar> &candidate) const {
        assert(candidate.size() == total_size &&
               "AllenCahnDenseSolver: candidate field size must equal "
               "prod(sizes).");
        const Eigen::VectorX<Scalar> &reference = history.front();
        const Scalar ref_norm = reference.norm();
        const Scalar diff_norm = (candidate - reference).norm();
        return (ref_norm > Scalar(0)) ? diff_norm / ref_norm : diff_norm;
    }

  private:
    /*! @brief The Allen-Cahn rhs \f$ F(f) = \kappa \, \Delta f + f - f^3 \f$ on
     * the flat field (the cube is elementwise). */
    [[nodiscard]] Eigen::VectorX<Scalar>
    rhs(const Eigen::VectorX<Scalar> &f) const {
        return kappa * laplacian(f) + f -
               f.array().cube().matrix();
    }

    /*!
     * @brief The periodic Laplacian
     * \f$ \Delta f = \sum_k L_k / h_k^2 \; f \f$ applied axis by axis, where
     * \f$ L_k \f$ is the central second-derivative stencil of order `order`
     * (the `stencil_weights` from `centralSecondDerivativeWeights`) with the
     * neighbours *wrapped around* modulo \f$ n_k \f$. Matches
     * `makePeriodicLaplacianOperator` exactly, including the periodic wrap that
     * replaces the acoustic Dirichlet truncation at the ends.
     *
     * Flat, first-mode-fastest layout: axis \f$ k \f$ has stride
     * \f$ \prod_{j<k} n_j \f$; the \f$ o \f$-th neighbour along axis \f$ k \f$
     * is at coordinate \f$ (\text{coord} \pm o) \bmod n_k \f$, whose flat index
     * moves by \f$ (\text{wrapped} - \text{coord}) \cdot \text{stride} \f$.
     */
    [[nodiscard]] Eigen::VectorX<Scalar>
    laplacian(const Eigen::VectorX<Scalar> &f) const {
        Eigen::VectorX<Scalar> out = Eigen::VectorX<Scalar>::Zero(total_size);
        const auto radius =
            static_cast<Eigen::Index>(stencil_weights.size()) - 1;
        Eigen::Index stride = 1;
        for (std::size_t k = 0; k < sizes.size(); ++k) {
            const Eigen::Index n = sizes[k];
            const Scalar inv_h2 = Scalar(1) / (spacings[k] * spacings[k]);
            for (Eigen::Index i = 0; i < total_size; ++i) {
                const Eigen::Index coord = (i / stride) % n;
                Scalar acc = stencil_weights[0] * f(i);
                for (Eigen::Index o = 1; o <= radius; ++o) {
                    const Scalar w = stencil_weights[static_cast<std::size_t>(o)];
                    const Eigen::Index lo = ((coord - o) % n + n) % n;
                    const Eigen::Index hi = (coord + o) % n;
                    acc += w * f(i + (lo - coord) * stride);
                    acc += w * f(i + (hi - coord) * stride);
                }
                out(i) += inv_h2 * acc;
            }
            stride *= n;
        }
        return out;
    }

    Scalar kappa;
    std::vector<Eigen::Index> sizes;
    std::vector<Scalar> spacings;
    Scheme<Scalar> scheme;
    Scalar dt;
    Scalar t = Scalar(0);
    Eigen::Index total_size = 1;
    // Symmetric central second-derivative weights (offsets 0..order/2), shared
    // by every axis; must match makePeriodicLaplacianOperator's stencil.
    std::vector<Scalar> stencil_weights;
    std::deque<Eigen::VectorX<Scalar>> history; // history[0] = f_n.
};

} // namespace MatSubset::Experiments

#endif // MAT_SUBSET_EXPERIMENTS_ALLEN_CAHN_DENSE_SOLVER_H
