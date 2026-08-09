// TT-leapfrog acoustic wave propagation in a layered medium, after
// Liu & Sacchi (GeoConvention 2025): 1/c^2 p_tt = lap(p) + s(x) f(t) on an
// n x n x n grid, a 3-layer speed model along depth, a Ricker source near the
// top, and the Cerjan sponge as the absorbing boundary. Snapshots go to
// ./snapshots as .vti files for ParaView.
//
// The dense (full-grid, rank-unlimited) solver runs the identical
// discretization as ground truth; the fixed-skeleton Solver and the
// AdaptiveSolver are compared against it by relative L2 residual per snapshot,
// so their reported errors are purely low-rank approximation error.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include <MatSubset/FrobeniusSelectionSelector.h>

#include <AcousticEquation/AcousticRhs.h>
#include <AcousticEquation/DenseSolver.h>
#include <AcousticEquation/FiniteDifference.h>
#include <TTCrossSolver/AdaptiveSolver.h>
#include <TTCrossSolver/SnapshotSaver.h>
#include <TTCrossSolver/Solver.h>
#include <TTCrossSolver/TensorTrain.h>
#include <TTCrossSolver/TensorTrainCore.h>

using Scalar = double;

using MatSubset::Experiments::AcousticRhs;
using MatSubset::Experiments::AdaptiveSolver;
using MatSubset::Experiments::DenseSolver;
using MatSubset::Experiments::Grid;
using MatSubset::Experiments::IdentityFieldMapper;
using MatSubset::Experiments::Scheme;
using MatSubset::Experiments::secondDerivativeSpectralFactor;
using MatSubset::Experiments::SnapshotSaver;
using MatSubset::Experiments::Solver;
using MatSubset::Experiments::TensorTrain;
using MatSubset::Experiments::TensorTrainCore;

namespace {

// A rank-1 3D train v0 x v1 x v2.
TensorTrain<Scalar> makeRank1(const Eigen::VectorX<Scalar> &v0,
                              const Eigen::VectorX<Scalar> &v1,
                              const Eigen::VectorX<Scalar> &v2) {
    std::vector<TensorTrainCore<Scalar>> cores;
    cores.emplace_back(Eigen::MatrixX<Scalar>(v0), v0.size());
    cores.emplace_back(Eigen::MatrixX<Scalar>(v1), v1.size());
    cores.emplace_back(Eigen::MatrixX<Scalar>(v2), v2.size());
    return TensorTrain<Scalar>(std::move(cores));
}

// The dense (flat, first-mode-fastest) field of a separable rank-1 tensor
// v0 x v1 x v2, matching TensorTrain::toDense()'s ordering.
Eigen::VectorX<Scalar> denseRank1(const Eigen::VectorX<Scalar> &v0,
                                  const Eigen::VectorX<Scalar> &v1,
                                  const Eigen::VectorX<Scalar> &v2) {
    const Eigen::Index n0 = v0.size(), n1 = v1.size(), n2 = v2.size();
    Eigen::VectorX<Scalar> field(n0 * n1 * n2);
    for (Eigen::Index k = 0; k < n2; ++k) {
        for (Eigen::Index j = 0; j < n1; ++j) {
            for (Eigen::Index i = 0; i < n0; ++i) {
                field(i + n0 * (j + n1 * k)) = v0(i) * v1(j) * v2(k);
            }
        }
    }
    return field;
}

Eigen::Index maxRank(const TensorTrain<Scalar> &train) {
    Eigen::Index r = 0;
    for (const Eigen::Index rank : train.ranks()) {
        r = std::max(r, rank);
    }
    return r;
}

// The interior bond ranks (r_1 .. r_{d-1}) as "r_1xr_2x...", dropping the
// trivial boundary ranks r_0 = r_d = 1. For a 3D train this is "r_1xr_2", so
// one can see whether the two bonds are equal.
std::string interiorRanks(const TensorTrain<Scalar> &train) {
    const std::vector<Eigen::Index> r = train.ranks();
    std::string out;
    for (std::size_t k = 1; k + 1 < r.size(); ++k) {
        out += (k > 1 ? "x" : "") + std::to_string(r[k]);
    }
    return out;
}

// Restricts a fine flat field on the nested grid ((n_c - 1) * stride + 1 points
// per axis) to the coarse n_c^3 nodes: coarse node g reads fine node stride * g
// (offset 0), the SparseFieldMapper decimation. Both fields are
// first-mode-fastest, so this is the exact coarse-node restriction with no
// interpolation (the nested grids share those nodes: coarse coord g*h_c ==
// fine coord (stride*g)*(h_c/stride)). Kept for re-enabling the refined
// reference (currently disabled — see main); [[maybe_unused]] until then.
[[maybe_unused]] Eigen::VectorX<Scalar>
restrictField(const Eigen::VectorX<Scalar> &fine, Eigen::Index n_coarse,
              Eigen::Index stride) {
    const Eigen::Index n_fine = (n_coarse - 1) * stride + 1;
    assert(fine.size() == n_fine * n_fine * n_fine &&
           "restrictField: fine field size must equal n_fine^3.");
    Eigen::VectorX<Scalar> coarse(n_coarse * n_coarse * n_coarse);
    for (Eigen::Index k = 0; k < n_coarse; ++k) {
        for (Eigen::Index j = 0; j < n_coarse; ++j) {
            for (Eigen::Index i = 0; i < n_coarse; ++i) {
                coarse(i + n_coarse * (j + n_coarse * k)) =
                    fine(stride * i +
                         n_fine * (stride * j + n_fine * stride * k));
            }
        }
    }
    return coarse;
}

// Relative L2 residual ||candidate - reference|| / ||reference||, with an
// absolute-norm fallback when the reference is (numerically) zero. Kept for the
// (currently disabled) refined reference; [[maybe_unused]] until re-enabled.
[[maybe_unused]] Scalar relativeL2(const Eigen::VectorX<Scalar> &candidate,
                                   const Eigen::VectorX<Scalar> &reference) {
    assert(candidate.size() == reference.size() &&
           "relativeL2: fields must have the same length.");
    const Scalar ref_norm = reference.norm();
    const Scalar diff_norm = (candidate - reference).norm();
    return (ref_norm > Scalar(0)) ? diff_norm / ref_norm : diff_norm;
}

} // namespace

int main(int argc, char **argv) {
    // Grid: n^3 points on the cube [0, extent]^3, z = depth.
    const Eigen::Index n = (argc > 1) ? std::atoi(argv[1]) : 128;
    const Scalar t_end = (argc > 2) ? std::atof(argv[2]) : Scalar(1);
    const Scalar rtol = (argc > 3) ? std::atof(argv[3]) : Scalar(1e-6);
    const Scalar width_factor = (argc > 4) ? std::atof(argv[4]) : Scalar(2);
    const Eigen::Index oversampling = (argc > 5) ? std::atoi(argv[5]) : 0;
    // Accuracy order of the central Laplacian stencil (even, >= 2). Higher
    // orders cut the numerical dispersion that otherwise drives the coarse and
    // refined grids out of phase over time. Shared by all three solvers.
    const int order = (argc > 7) ? std::atoi(argv[7]) : 8;

    const auto num_samples = [width_factor, oversampling](Eigen::Index rank,
                                                          Eigen::Index) {
        return static_cast<Eigen::Index>(
                   std::ceil(width_factor * static_cast<Scalar>(rank))) +
               oversampling;
    };
    const Scalar extent = Scalar(2000);                   // m
    const Scalar h = extent / static_cast<Scalar>(n - 1); // m

    const std::vector<Eigen::Index> sizes{n, n, n};
    const std::vector<Scalar> spacings{h, h, h};
    const Grid<Scalar> grid(sizes, {Scalar(0), Scalar(0), Scalar(0)},
                            {extent, extent, extent});

    // 3-layer medium: c depends on depth only, so the speed train is rank 1
    // (ones x ones x c_z). Source: a separable Gaussian ball (rank 1), centered
    // in x, y and near the top in z, two cells wide (relative to the coarse h,
    // so the same physical source on every grid). Both are functions of the
    // physical coordinate, so they are rebuilt per axis on whichever grid asks.
    const Scalar c_top = Scalar(1500);    // m/s
    const Scalar c_middle = Scalar(2000); // m/s
    const Scalar c_bottom = Scalar(2500); // m/s
    const Scalar sigma = Scalar(2) * h;
    const std::vector<Scalar> source_at{extent / Scalar(2), extent / Scalar(2),
                                        extent / Scalar(5)};

    // The medium's per-axis speed factor c_k(x) on a grid of `m` points: ones
    // along x, y and the layered profile along z (axis 2).
    const auto speedAxis = [&](std::size_t k, Eigen::Index m) {
        const Grid<Scalar> g({m, m, m}, {Scalar(0), Scalar(0), Scalar(0)},
                             {extent, extent, extent});
        Eigen::VectorX<Scalar> v(m);
        for (Eigen::Index i = 0; i < m; ++i) {
            if (k != 2) {
                v(i) = Scalar(1);
            } else {
                const Scalar z = g.coordinate(2, i);
                v(i) = (z < extent / Scalar(3))               ? c_top
                       : (z < Scalar(2) * extent / Scalar(3)) ? c_middle
                                                              : c_bottom;
            }
        }
        return v;
    };
    // The source's per-axis Gaussian factor s_k(x) on a grid of `m` points.
    const auto sourceAxis = [&](std::size_t k, Eigen::Index m) {
        const Grid<Scalar> g({m, m, m}, {Scalar(0), Scalar(0), Scalar(0)},
                             {extent, extent, extent});
        Eigen::VectorX<Scalar> v(m);
        for (Eigen::Index i = 0; i < m; ++i) {
            const Scalar d = g.coordinate(k, i) - source_at[k];
            v(i) = std::exp(-d * d / (Scalar(2) * sigma * sigma));
        }
        return v;
    };

    // Coarse (grid-n) speed and source axes, shared by the TT solvers and the
    // same-grid dense solver.
    const Eigen::VectorX<Scalar> c_z = speedAxis(2, n);
    const Eigen::VectorX<Scalar> ones = Eigen::VectorX<Scalar>::Ones(n);
    const Eigen::VectorX<Scalar> src0 = sourceAxis(0, n);
    const Eigen::VectorX<Scalar> src1 = sourceAxis(1, n);
    const Eigen::VectorX<Scalar> src2 = sourceAxis(2, n);

    // Ricker wavelet, f0 = 10 Hz as in the layered experiment of the report,
    // delayed so it starts from (numerical) zero.
    const Scalar f0 = Scalar(10); // Hz
    const Scalar t0 = Scalar(1.5) / f0;
    const auto ricker = [f0, t0](Scalar t) {
        const Scalar a = Scalar(M_PI) * f0 * (t - t0);
        const Scalar a2 = a * a;
        return (Scalar(1) - Scalar(2) * a2) * std::exp(-a2);
    };

    // CFL for leapfrog on the order-`order` Laplacian in d = 3 dimensions:
    // stability needs dt <= 2 / sqrt(lambda_max) with the Laplacian spectral
    // radius lambda_max = d * S / h^2, S the per-axis spectral factor of the
    // stencil (4 at order 2, larger at higher orders). We take 1/2 of the
    // limit. At order 2, S = 4, this is the classic 0.5 * h / (c sqrt(3)).
    const Scalar spectral_factor =
        secondDerivativeSpectralFactor<Scalar>(order);
    const Scalar dt =
        Scalar(0.5) * Scalar(2) * h /
        (c_bottom * std::sqrt(Scalar(sizes.size()) * spectral_factor));
    const int n_steps = static_cast<int>(std::ceil(t_end / dt));
    const int save_every = 10; // std::max(1, n_steps / 60);

    std::cout << "grid " << n << "^3, h = " << h << " m, dt = " << dt << " s, "
              << n_steps << " steps to t = " << t_end << " s\n"
              << "rtol = " << rtol << ", width policy = ceil(" << width_factor
              << " * r) + " << oversampling << ", stencil order = " << order
              << '\n';

    // The medium, source and forcing shared by all three solvers, in the two
    // representations they need: TT trains for the fiber solvers, flat
    // first-mode-fastest fields for the dense reference.
    const auto makeSpeed = [&] { return makeRank1(ones, ones, c_z); };
    const auto makeSource = [&] { return makeRank1(src0, src1, src2); };
    const Scheme<Scalar> scheme = Scheme<Scalar>::leapfrogSecondOrder();

    // No absorbing boundary for now: the walls are the Laplacian's homogeneous
    // Dirichlet ends, so the wavefield reflects off them. The BC is not what
    // this experiment studies; see makeCerjanMask / MaskBoundaryCondition to
    // re-enable a Cerjan sponge.

    // The wavefield starts at rest as exact zero states. The first fiber steps
    // run in exact TT arithmetic (warm-up below), so no skeleton is ever
    // selected from a structureless state.
    std::vector<TensorTrain<Scalar>> initial_history_solver;
    initial_history_solver.push_back(makeRank1(
        Eigen::VectorX<Scalar>::Zero(n), Eigen::VectorX<Scalar>::Zero(n),
        Eigen::VectorX<Scalar>::Zero(n))); // p_{-1}
    initial_history_solver.push_back(makeRank1(
        Eigen::VectorX<Scalar>::Zero(n), Eigen::VectorX<Scalar>::Zero(n),
        Eigen::VectorX<Scalar>::Zero(n))); // p_0
    std::vector<TensorTrain<Scalar>> initial_history_adaptive =
        initial_history_solver;

    const Eigen::Index total = n * n * n;
    std::vector<Eigen::VectorX<Scalar>> initial_history_dense;
    initial_history_dense.push_back(Eigen::VectorX<Scalar>::Zero(total));
    initial_history_dense.push_back(Eigen::VectorX<Scalar>::Zero(total));

    // The 2x-refined reference (a nested-grid dense solve at dt/2) is disabled:
    // at the resolutions this study runs (e.g. 256^3, so a 511^3 fine solve at
    // half the step) it is prohibitively slow, and it measured discretization
    // error, which is not the subject here. The same-grid dense solver above
    // still isolates the low-rank error; see restrictField / relativeL2 in the
    // helper namespace to re-enable it on a small grid.

    // Deterministic greedy selection. FrobeniusSelectionSelector picks columns
    // that minimize the pseudo-inverse Frobenius norm via rank-1 updates:
    // O(n r^3) per bond against DerandomizedVolumeSelector's O(n r^4), an
    // r-fold win that dominates the step at the ranks this solve reaches
    // (~5x at rank 90).
    const Scalar atol = Scalar(0);

    // Warm-up: exact TT steps (rank-inflated combos + compress) until the
    // wavefield has structure worth selecting a skeleton from.
    const int warmup_steps = (argc > 6) ? std::atoi(argv[6]) : 10;

    // Same-grid dense solver: full-grid, no low-rank compression, on the same
    // grid / stencil / scheme / dt as the TT solvers. Isolates the low-rank
    // approximation error.
    DenseSolver<Scalar> dense_solver(
        denseRank1(ones, ones, c_z), denseRank1(src0, src1, src2), ricker,
        sizes, spacings, scheme, dt, std::move(initial_history_dense), order);

    // Fixed-skeleton solver.
    Solver<Scalar> solver(
        std::move(initial_history_solver),
        std::make_unique<AcousticRhs<Scalar>>(makeSpeed(), makeSource(), ricker,
                                              sizes, spacings, order),
        scheme, dt, std::make_unique<MatSubset::FrobeniusSelectionSelector<Scalar>>(),
        atol, rtol, num_samples, /*boundary_condition=*/nullptr, warmup_steps);

    // Adaptive solver: re-selects the skeleton against each stage combo.
    AdaptiveSolver<Scalar> adaptive_solver(
        std::move(initial_history_adaptive),
        std::make_unique<AcousticRhs<Scalar>>(makeSpeed(), makeSource(), ricker,
                                              sizes, spacings, order),
        scheme, dt, std::make_unique<MatSubset::FrobeniusSelectionSelector<Scalar>>(),
        atol, rtol, num_samples, /*boundary_condition=*/nullptr, warmup_steps);

    std::vector<
        std::unique_ptr<MatSubset::Experiments::FieldMapperBase<Scalar>>>
        fields;
    fields.push_back(std::make_unique<IdentityFieldMapper<Scalar>>("pressure"));
    SnapshotSaver<Scalar> saver(
        grid, std::move(fields), "out/snapshots", "wavefield",
        MatSubset::Experiments::StoragePrecision::Float);

    saver.save(adaptive_solver.getState(), adaptive_solver.time());
    const auto solve_start = std::chrono::steady_clock::now();
    for (int step = 1; step <= n_steps; ++step) {
        dense_solver.step();
        solver.step();
        adaptive_solver.step();

        if (step % save_every == 0 || step == n_steps) {
            const std::string path =
                 saver.save(adaptive_solver.getState(), adaptive_solver.time());

            // Same-grid dense residuals: each TT state reconstructed to a full
            // field (toDense() shares the dense field's first-mode-fastest
            // layout) and differenced node-for-node. Isolates the low-rank
            // approximation error.
            const Scalar res_solver =
                dense_solver.relativeResidual(solver.getState().toDense());
            const Scalar res_adaptive = dense_solver.relativeResidual(
                adaptive_solver.getState().toDense());

            const auto elapsed =
                std::chrono::duration<Scalar>(std::chrono::steady_clock::now() -
                                              solve_start)
                    .count();
            std::cout << "step " << step << " / " << n_steps
                      << ", t = " << adaptive_solver.time()
                      << " s, ranks: solver = " << maxRank(solver.getState())
                      << " (" << interiorRanks(solver.getState()) << ")"
                      << ", adaptive = " << maxRank(adaptive_solver.getState())
                      << " (" << interiorRanks(adaptive_solver.getState()) << ")"
                      << ", low-rank err: solver = " << res_solver
                      << ", adaptive = " << res_adaptive
                      << ", elapsed = " << elapsed << " s (" << elapsed / step
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
