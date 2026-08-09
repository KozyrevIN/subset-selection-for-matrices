#ifndef MAT_SUBSET_EXPERIMENTS_DENSE_SOLVER_H
#define MAT_SUBSET_EXPERIMENTS_DENSE_SOLVER_H

#include <cassert>    // For assert
#include <cstddef>    // For std::size_t
#include <deque>      // For std::deque
#include <functional> // For std::function
#include <utility>    // For std::move
#include <vector>     // For std::vector

#include <Eigen/Core> // For Eigen::VectorX, Eigen::Index

#include "AcousticEquation/FiniteDifference.h" // For centralSecondDerivat...
#include "TTCrossSolver/SolverBase.h"          // For Scheme

namespace MatSubset::Experiments {

/*!
 * @brief Full-grid ("dense") reference solver for the acoustic wave equation
 * \f$ \frac{1}{c(x)^2} \partial_t^2 p = \Delta p + s(x) f(t) \f$, stepping the
 * unmodified field with no low-rank compression at all.
 * @tparam Scalar The underlying scalar type (e.g. `float`, `double`).
 *
 * The rank-unlimited ground truth for the TT solvers: it runs the *same*
 * discretization they do — the second-order Dirichlet Laplacian stencil
 * \f$ \mathrm{tridiag}(1, -2, 1) / h_k^2 \f$ summed over the axes, the same
 * medium \f$ c(x) \f$, the same separable forcing \f$ s(x) f(t) \f$, and the
 * same time-stepping `Scheme` — but stores the field as a plain flat array
 * and never truncates. Its state is therefore the exact (up to round-off)
 * solution of the discrete problem on this grid, so the difference between it
 * and a TT solver's state is purely that solver's low-rank approximation error.
 *
 * The state layout matches `TensorTrain::toDense()`: a flat vector of length
 * \f$ \prod_k n_k \f$ in first-mode-fastest (column-major) order, so a TT
 * state's `toDense()` and this solver's `field()` are directly comparable
 * node-for-node (see `relativeResidual`).
 *
 * The rhs is baked in — this is not the generic `RhsBase` chassis — because a
 * dense reference only needs to exist for exactly the acoustic experiment the
 * TT solvers are being checked against.
 */
template <typename Scalar> class DenseSolver {
  public:
    /*!
     * @brief Constructs the solver from the medium, the source and the grid.
     * @param speed The sound speed \f$ c(x) \f$ as a flat field
     * (first-mode-fastest, length \f$ \prod_k n_k \f$).
     * @param source_spatial The spatial source factor \f$ s(x) \f$ as a flat
     * field, same layout and length.
     * @param source_time The time envelope \f$ f(t) \f$.
     * @param sizes The grid points per dimension \f$ (n_1, \dots, n_d) \f$.
     * @param spacings The grid spacings \f$ (h_1, \dots, h_d) \f$ (for the
     * Laplacian stencil).
     * @param scheme The time-stepping scheme (e.g.
     * `Scheme::leapfrogSecondOrder()`).
     * @param dt The time step.
     * @param initial_history The `scheme.history` initial states as flat
     * fields, in chronological order, most recent last (e.g.
     * \f$ \{p_{-1}, p_0\} \f$ for leapfrog); each of length
     * \f$ \prod_k n_k \f$.
     * @param order The (even) accuracy order of the Laplacian stencil
     * (default 2); must match the `AcousticRhs` order of the TT solvers being
     * compared for the same-grid dense residual to isolate the low-rank error.
     */
    DenseSolver(Eigen::VectorX<Scalar> speed,
                Eigen::VectorX<Scalar> source_spatial,
                std::function<Scalar(Scalar)> source_time,
                std::vector<Eigen::Index> sizes, std::vector<Scalar> spacings,
                Scheme<Scalar> scheme, Scalar dt,
                std::vector<Eigen::VectorX<Scalar>> initial_history,
                int order = 2)
        : speed_squared(speed.array().square().matrix()),
          source_spatial(std::move(source_spatial)),
          source_time(std::move(source_time)), sizes(std::move(sizes)),
          spacings(std::move(spacings)), scheme(std::move(scheme)), dt(dt),
          stencil_weights(centralSecondDerivativeWeights<Scalar>(order)) {
        assert(this->source_time && "DenseSolver: null time envelope.");
        assert(!this->scheme.stages.empty() &&
               "DenseSolver: the scheme needs at least one stage.");
        assert((this->scheme.time_order == 1 || this->scheme.time_order == 2) &&
               "DenseSolver: time_order must be 1 or 2.");
        assert(this->sizes.size() == this->spacings.size() &&
               "DenseSolver: one spacing per dimension.");

        total_size = 1;
        for (const Eigen::Index n : this->sizes) {
            assert(n >= 1 && "DenseSolver: each axis needs >= 1 point.");
            total_size *= n;
        }
        assert(speed.size() == total_size &&
               "DenseSolver: speed field size must equal prod(sizes).");
        assert(this->source_spatial.size() == total_size &&
               "DenseSolver: source field size must equal prod(sizes).");
        assert(initial_history.size() == this->scheme.history &&
               "DenseSolver: initial history must provide exactly "
               "scheme.history states.");

        // Most recent state last on input, front of the deque internally
        // (history[0] = p_n), matching SolverBase's convention.
        for (auto it = initial_history.rbegin(); it != initial_history.rend();
             ++it) {
            assert(it->size() == total_size &&
                   "DenseSolver: each initial state size must equal "
                   "prod(sizes).");
            history.push_back(std::move(*it));
        }
    }

    /*! @brief Advances the field by one time step. */
    void step() {
        const Scalar dtp = (scheme.time_order == 2) ? dt * dt : dt;

        // Stage recursion Y^(0) = p_n, exactly as SolverBase but on flat
        // fields with no truncation.
        Eigen::VectorX<Scalar> stage_state = history.front();
        for (std::size_t j = 0; j < scheme.stages.size(); ++j) {
            const auto &stage = scheme.stages[j];

            // combo = sum_m alpha_m p_{n-m} + gamma dt^p F(Y^(j-1), t + theta dt).
            Eigen::VectorX<Scalar> combo =
                (stage.rhs_weight * dtp) *
                rhs(stage_state, t + stage.rhs_time_offset * dt);
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

    /*! @brief The current field \f$ p_n \f$ as a flat vector, first-mode-fastest
     * (comparable node-for-node with `TensorTrain::toDense()`). */
    [[nodiscard]] const Eigen::VectorX<Scalar> &field() const {
        return history.front();
    }

    /*! @brief The current time \f$ t_n \f$ (starts at 0). */
    [[nodiscard]] Scalar time() const { return t; }

    /*!
     * @brief The relative \f$ L^2 \f$ residual of a candidate field against
     * this (reference) field: \f$ \lVert p - p_{\text{ref}} \rVert_2 /
     * \lVert p_{\text{ref}} \rVert_2 \f$.
     * @param candidate A flat field in the same layout and of the same length
     * as `field()` (e.g. a TT state's `toDense()`).
     * @return The relative error; falls back to the absolute norm of the
     * difference when the reference field is (numerically) zero.
     */
    [[nodiscard]] Scalar
    relativeResidual(const Eigen::VectorX<Scalar> &candidate) const {
        assert(candidate.size() == total_size &&
               "DenseSolver: candidate field size must equal prod(sizes).");
        const Eigen::VectorX<Scalar> &reference = history.front();
        const Scalar ref_norm = reference.norm();
        const Scalar diff_norm = (candidate - reference).norm();
        return (ref_norm > Scalar(0)) ? diff_norm / ref_norm : diff_norm;
    }

  private:
    /*! @brief The acoustic rhs \f$ F(p, t) = c^2 \odot (\Delta p + s(x) f(t))
     * \f$ on the flat field. */
    [[nodiscard]] Eigen::VectorX<Scalar> rhs(const Eigen::VectorX<Scalar> &p,
                                             Scalar time_value) const {
        Eigen::VectorX<Scalar> forced =
            laplacian(p) + source_time(time_value) * source_spatial;
        return speed_squared.cwiseProduct(forced);
    }

    /*!
     * @brief The Dirichlet Laplacian
     * \f$ \Delta p = \sum_k L_k / h_k^2 \; p \f$ applied axis by axis, where
     * \f$ L_k \f$ is the central second-derivative stencil of order `order`
     * (the `stencil_weights` from `centralSecondDerivativeWeights`). Matches
     * `makeLaplacianOperator` exactly, including the "drop any term reaching
     * outside \f$ [0, n-1] \f$" homogeneous Dirichlet truncation at the ends.
     *
     * Flat, first-mode-fastest layout: axis \f$ k \f$ has stride
     * \f$ \prod_{j<k} n_j \f$, so the \f$ j \f$-th neighbour along axis \f$ k \f$
     * is `j * stride` entries away; the truncation checks keep every stencil
     * term inside the axis's own slab.
     */
    [[nodiscard]] Eigen::VectorX<Scalar>
    laplacian(const Eigen::VectorX<Scalar> &p) const {
        Eigen::VectorX<Scalar> out = Eigen::VectorX<Scalar>::Zero(total_size);
        const auto radius =
            static_cast<Eigen::Index>(stencil_weights.size()) - 1;
        Eigen::Index stride = 1;
        for (std::size_t k = 0; k < sizes.size(); ++k) {
            const Eigen::Index n = sizes[k];
            const Scalar inv_h2 = Scalar(1) / (spacings[k] * spacings[k]);
            // For every flat index i, its position along axis k is
            // (i / stride) % n; the o-th neighbour along the axis moves i by
            // +/- o * stride, dropped when it leaves [0, n-1].
            for (Eigen::Index i = 0; i < total_size; ++i) {
                const Eigen::Index coord = (i / stride) % n;
                Scalar acc = stencil_weights[0] * p(i);
                for (Eigen::Index o = 1; o <= radius; ++o) {
                    const Scalar w = stencil_weights[static_cast<std::size_t>(o)];
                    if (coord - o >= 0) {
                        acc += w * p(i - o * stride);
                    }
                    if (coord + o < n) {
                        acc += w * p(i + o * stride);
                    }
                }
                out(i) += inv_h2 * acc;
            }
            stride *= n;
        }
        return out;
    }

    Eigen::VectorX<Scalar> speed_squared; // c^2 as a flat field.
    Eigen::VectorX<Scalar> source_spatial;
    std::function<Scalar(Scalar)> source_time;
    std::vector<Eigen::Index> sizes;
    std::vector<Scalar> spacings;
    Scheme<Scalar> scheme;
    Scalar dt;
    Scalar t = Scalar(0);
    Eigen::Index total_size = 1;
    // Symmetric central second-derivative weights (offsets 0..order/2), shared
    // by every axis; must match makeLaplacianOperator's stencil.
    std::vector<Scalar> stencil_weights;
    std::deque<Eigen::VectorX<Scalar>> history; // history[0] = p_n.
};

} // namespace MatSubset::Experiments

#endif // MAT_SUBSET_EXPERIMENTS_DENSE_SOLVER_H
