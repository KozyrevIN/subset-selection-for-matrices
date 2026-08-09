// TT low-rank integration of the 3D Allen-Cahn equation
//   df/dt = kappa * lap(f) + f - f^3   on the periodic cube [0, 2*pi]^3,
// with kappa = 0.1 and the non-separable initial condition
//   f0 = g(x1,x2,x3) - g(2x1,x2,x3) + g(x1,2x2,x3) - g(x1,x2,2x3)
// of the report (see AllenCahnInitialCondition.h for g). Snapshots go to
// ./out/snapshots as .vti files for ParaView.
//
// The domain is periodic, so the grid is the interior lattice x_k = 2*pi*k/n
// (k = 0 .. n-1, endpoints excluded) and the Laplacian wraps around. The
// equation is first order in time and stiff in the diffusion term, so an
// explicit SSP-RK3 step is taken under the diffusion CFL dt ~ h^2 / kappa.
//
// The dense (full-grid, rank-unlimited) solver runs the identical
// discretization as ground truth; the fixed-skeleton Solver and the
// AdaptiveSolver are compared against it by relative L2 residual per snapshot,
// so their reported errors are purely low-rank approximation error.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include <MatSubset/FrobeniusSelectionSelector.h>

#include <AllenCahnEquation/AllenCahnDenseSolver.h>
#include <AllenCahnEquation/AllenCahnInitialCondition.h>
#include <AllenCahnEquation/AllenCahnRhs.h>
#include <TTCrossSolver/AdaptiveSolver.h>
#include <TTCrossSolver/SnapshotSaver.h>
#include <TTCrossSolver/Solver.h>
#include <TTCrossSolver/TensorTrain.h>

using Scalar = double;

using MatSubset::Experiments::AdaptiveSolver;
using MatSubset::Experiments::AllenCahnDenseSolver;
using MatSubset::Experiments::AllenCahnRhs;
using MatSubset::Experiments::allenCahnInitial;
using MatSubset::Experiments::denseFieldFromFunction;
using MatSubset::Experiments::Grid;
using MatSubset::Experiments::IdentityFieldMapper;
using MatSubset::Experiments::Scheme;
using MatSubset::Experiments::secondDerivativeSpectralFactor;
using MatSubset::Experiments::SnapshotSaver;
using MatSubset::Experiments::Solver;
using MatSubset::Experiments::TensorTrain;
using MatSubset::Experiments::ttFromDenseTensor;

namespace {

Eigen::Index maxRank(const TensorTrain<Scalar> &train) {
    Eigen::Index r = 0;
    for (const Eigen::Index rank : train.ranks()) {
        r = std::max(r, rank);
    }
    return r;
}

// The interior bond ranks (r_1 .. r_{d-1}) as "r_1xr_2x...", dropping the
// trivial boundary ranks r_0 = r_d = 1.
std::string interiorRanks(const TensorTrain<Scalar> &train) {
    const std::vector<Eigen::Index> r = train.ranks();
    std::string out;
    for (std::size_t k = 1; k + 1 < r.size(); ++k) {
        out += (k > 1 ? "x" : "") + std::to_string(r[k]);
    }
    return out;
}

} // namespace

int main(int argc, char **argv) {
    // Periodic grid: n^3 interior nodes x_k = 2*pi*k/n on the cube [0, 2*pi]^3,
    // endpoints excluded (so the csc(-x/2) singularities of the IC are never
    // sampled). The spacing is h = 2*pi/n.
    const Eigen::Index n = (argc > 1) ? std::atoi(argv[1]) : 128;
    const Scalar t_end = (argc > 2) ? std::atof(argv[2]) : Scalar(10);
    const Scalar rtol = (argc > 3) ? std::atof(argv[3]) : Scalar(1e-4);
    const Scalar width_factor = (argc > 4) ? std::atof(argv[4]) : Scalar(2);
    const Eigen::Index oversampling = (argc > 5) ? std::atoi(argv[5]) : 0;
    // Warm-up (exact TT steps) defaults to 0: unlike the acoustic experiment,
    // whose initial state is exactly zero and thus structureless, the Allen-Cahn
    // initial condition is compressed from a non-separable function and already
    // carries plenty of rank structure for the selector to bootstrap from. Warm
    // up is also costly here — evaluateTrain cubes the state in exact TT
    // arithmetic (rank grows as r^3), which is prohibitive at a near-full-rank
    // initial state — so it is opt-in via the argument rather than a default.
    const int warmup_steps = (argc > 6) ? std::atoi(argv[6]) : 0;
    // Accuracy order of the central (periodic) Laplacian stencil (even, >= 2),
    // shared by all three solvers.
    const int order = (argc > 7) ? std::atoi(argv[7]) : 4;
    // Whether to run the dense (full-grid) reference solver alongside the TT
    // solvers. It dominates the wall time at fine grids (O(n^3) per stage), so
    // long production runs pass 0 here and give up the per-snapshot low-rank
    // residuals in exchange.
    const bool with_dense = (argc > 8) ? (std::atoi(argv[8]) != 0) : true;

    const Scalar kappa = Scalar(0.1);

    const auto num_samples = [width_factor, oversampling](Eigen::Index rank,
                                                          Eigen::Index) {
        return static_cast<Eigen::Index>(
                   std::ceil(width_factor * static_cast<Scalar>(rank))) +
               oversampling;
    };

    // Periodic interior lattice: n points spanning [0, 2*pi) with spacing
    // h = 2*pi/n. The grid's upper corner is 2*pi - h so that node n-1 sits one
    // spacing short of 2*pi (the periodic image of node 0).
    const Scalar two_pi = Scalar(2) * Scalar(M_PI);
    const Scalar h = two_pi / static_cast<Scalar>(n);
    const std::vector<Eigen::Index> sizes{n, n, n};
    const std::vector<Scalar> spacings{h, h, h};
    const Grid<Scalar> grid(sizes, {Scalar(0), Scalar(0), Scalar(0)},
                            {two_pi - h, two_pi - h, two_pi - h});

    // CFL for an explicit step on the order-`order` periodic Laplacian in
    // d = 3 dimensions. The diffusion operator kappa * lap has spectral radius
    // kappa * d * S / h^2 (S the per-axis stencil spectral factor); an explicit
    // scheme needs dt below O(1 / that). We take a conservative 1/4 of the
    // pure-diffusion Euler limit 2 / (kappa d S / h^2). The reaction f - f^3 is
    // non-stiff on the order-1 timescale and does not tighten this.
    const Scalar spectral_factor =
        secondDerivativeSpectralFactor<Scalar>(order);
    const Scalar dt =
        Scalar(0.25) * Scalar(2) /
        (kappa * Scalar(sizes.size()) * spectral_factor / (h * h));
    const int n_steps = static_cast<int>(std::ceil(t_end / dt));
    const int save_every = std::max(1, n_steps / 60);

    std::cout << "grid " << n << "^3 (periodic), h = " << h
              << ", dt = " << dt << ", " << n_steps << " steps to t = " << t_end
              << "\nkappa = " << kappa << ", rtol = " << rtol
              << ", width policy = ceil(" << width_factor << " * r) + "
              << oversampling << ", stencil order = " << order
              << ", warmup = " << warmup_steps
              << ", dense reference = " << (with_dense ? "on" : "off") << '\n';

    // The non-separable initial condition f0, sampled on the grid: as a flat
    // field for the dense reference and TT-SVD-compressed to a train for the
    // fiber solvers. Compress at the solver's own rtol: every step truncates at
    // rtol anyway, so a tighter initial train only inflates the starting ranks
    // (near full rank at fine grids — singular values of this f0 decay slowly
    // below ~1e-3) and the first steps' cost with it, without improving the
    // reported errors, which are measured against the dense reference.
    const Eigen::VectorX<Scalar> f0_dense = denseFieldFromFunction<Scalar>(
        grid, [](Scalar x1, Scalar x2, Scalar x3) {
            return allenCahnInitial<Scalar>(x1, x2, x3);
        });
    const TensorTrain<Scalar> f0_train =
        ttFromDenseTensor<Scalar>(f0_dense, sizes, rtol);
    std::cout << "initial state: TT ranks " << interiorRanks(f0_train)
              << " (max " << maxRank(f0_train) << ")\n";

    // SSP-RK3, the explicit first-order-in-time scheme for this stiff-diffusion
    // reaction-diffusion problem (lowStorageRK{1/3, 1/2, 1}).
    const Scheme<Scalar> scheme = Scheme<Scalar>::lowStorageRK(
        {Scalar(1) / Scalar(3), Scalar(1) / Scalar(2), Scalar(1)});

    const Scalar atol = Scalar(0);

    // Same-grid dense reference: full-grid, no compression, same grid / stencil
    // / scheme / dt as the TT solvers. Isolates the low-rank error. Optional —
    // it dominates the wall time at fine grids — hence the optional wrapper.
    std::optional<AllenCahnDenseSolver<Scalar>> dense_solver;
    if (with_dense) {
        std::vector<Eigen::VectorX<Scalar>> initial_history_dense{f0_dense};
        dense_solver.emplace(kappa, sizes, spacings, scheme, dt,
                             std::move(initial_history_dense), order);
    }

    // Fixed-skeleton solver.
    std::vector<TensorTrain<Scalar>> initial_history_solver{f0_train};
    Solver<Scalar> solver(
        std::move(initial_history_solver),
        std::make_unique<AllenCahnRhs<Scalar>>(kappa, sizes, spacings, order),
        scheme, dt,
        std::make_unique<MatSubset::FrobeniusSelectionSelector<Scalar>>(), atol,
        rtol, num_samples, /*boundary_condition=*/nullptr, warmup_steps);

    // Adaptive solver: re-selects the skeleton against each stage combo.
    std::vector<TensorTrain<Scalar>> initial_history_adaptive{f0_train};
    AdaptiveSolver<Scalar> adaptive_solver(
        std::move(initial_history_adaptive),
        std::make_unique<AllenCahnRhs<Scalar>>(kappa, sizes, spacings, order),
        scheme, dt,
        std::make_unique<MatSubset::FrobeniusSelectionSelector<Scalar>>(), atol,
        rtol, num_samples, /*boundary_condition=*/nullptr, warmup_steps);

    std::vector<
        std::unique_ptr<MatSubset::Experiments::FieldMapperBase<Scalar>>>
        fields;
    fields.push_back(std::make_unique<IdentityFieldMapper<Scalar>>("phase"));
    SnapshotSaver<Scalar> saver(
        grid, std::move(fields), "out/snapshots", "allen_cahn",
        MatSubset::Experiments::StoragePrecision::Float);

    saver.save(adaptive_solver.getState(), adaptive_solver.time());
    const auto solve_start = std::chrono::steady_clock::now();
    for (int step = 1; step <= n_steps; ++step) {
        if (dense_solver) {
            dense_solver->step();
        }
        solver.step();
        adaptive_solver.step();

        if (step % save_every == 0 || step == n_steps) {
            saver.save(adaptive_solver.getState(), adaptive_solver.time());

            const auto elapsed =
                std::chrono::duration<Scalar>(std::chrono::steady_clock::now() -
                                              solve_start)
                    .count();
            std::cout << "step " << step << " / " << n_steps
                      << ", t = " << adaptive_solver.time()
                      << ", ranks: solver = " << maxRank(solver.getState())
                      << " (" << interiorRanks(solver.getState()) << ")"
                      << ", adaptive = " << maxRank(adaptive_solver.getState())
                      << " (" << interiorRanks(adaptive_solver.getState())
                      << ")";
            if (dense_solver) {
                // Same-grid dense residuals: each TT state reconstructed to a
                // full field (toDense() shares the dense field's
                // first-mode-fastest layout) and differenced node-for-node.
                // Isolates the low-rank approximation error.
                const Scalar res_solver = dense_solver->relativeResidual(
                    solver.getState().toDense());
                const Scalar res_adaptive = dense_solver->relativeResidual(
                    adaptive_solver.getState().toDense());
                std::cout << ", low-rank err: solver = " << res_solver
                          << ", adaptive = " << res_adaptive;
            }
            std::cout << ", elapsed = " << elapsed << " s (" << elapsed / step
                      << " s/step)\n";
        }
    }
    const auto solve_seconds =
        std::chrono::duration<Scalar>(std::chrono::steady_clock::now() -
                                      solve_start)
            .count();
    std::cout << "solve took " << solve_seconds << " s for " << n_steps
              << " steps (" << solve_seconds / n_steps << " s/step)\n";

    return 0;
}
