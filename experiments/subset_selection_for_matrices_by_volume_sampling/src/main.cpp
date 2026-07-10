// TT-leapfrog acoustic wave propagation in a layered medium, after
// Liu & Sacchi (GeoConvention 2025): 1/c^2 p_tt = lap(p) + s(x) f(t) on an
// n x n x n grid, a 3-layer speed model along depth, a Ricker source near the
// top, and the Cerjan sponge as the absorbing boundary. Snapshots go to
// ./snapshots as .vti files for ParaView.

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include <MatSubset/ForwardIterativeVolumeSamplingSelector.h>

#include <AcousticEquation/AcousticRhs.h>
#include <TTCrossSolver/SnapshotSaver.h>
#include <TTCrossSolver/Solver.h>
#include <TTCrossSolver/TensorTrain.h>
#include <TTCrossSolver/TensorTrainCore.h>

using Scalar = double;

using MatSubset::Experiments::AcousticRhs;
using MatSubset::Experiments::Grid;
using MatSubset::Experiments::IdentityFieldMapper;
using MatSubset::Experiments::makeCerjanMask;
using MatSubset::Experiments::MaskBoundaryCondition;
using MatSubset::Experiments::Scheme;
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

Eigen::Index maxRank(const TensorTrain<Scalar> &train) {
    Eigen::Index r = 0;
    for (const Eigen::Index rank : train.ranks()) {
        r = std::max(r, rank);
    }
    return r;
}

} // namespace

int main(int argc, char **argv) {
    // Grid: n^3 points on the cube [0, extent]^3, z = depth.
    const Eigen::Index n = (argc > 1) ? std::atoi(argv[1]) : 64;
    const Scalar t_end = (argc > 2) ? std::atof(argv[2]) : Scalar(1);
    // Truncation tolerance and the skeleton width policy
    // ceil(factor * rank) + oversampling. The rank-proportional slack keeps
    // the per-step cross rebuild from silently truncating the leapfrog combo
    // (whose numerical rank exceeds the state's) as the resolution grows; a
    // purely constant budget works at one grid size and destabilizes at the
    // next. Do not starve the solver either: a loose rtol (1e-3) pins the
    // rank near 4, which cannot represent a spherical wavefront (O(1) error,
    // axis-aligned lobes).
    const Scalar rtol = (argc > 3) ? std::atof(argv[3]) : Scalar(1e-6);
    const Scalar width_factor = (argc > 4) ? std::atof(argv[4]) : Scalar(1.2);
    const Eigen::Index oversampling = (argc > 5) ? std::atoi(argv[5]) : 4;
    const auto num_samples = [width_factor,
                              oversampling](Eigen::Index rank, Eigen::Index) {
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
    // (ones x ones x c_z).
    const Scalar c_top = Scalar(1500);    // m/s
    const Scalar c_middle = Scalar(2000); // m/s
    const Scalar c_bottom = Scalar(2500); // m/s
    Eigen::VectorX<Scalar> c_z(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        const Scalar z = grid.coordinate(2, i);
        c_z(i) = (z < extent / Scalar(3))               ? c_top
                 : (z < Scalar(2) * extent / Scalar(3)) ? c_middle
                                                        : c_bottom;
    }
    const Eigen::VectorX<Scalar> ones = Eigen::VectorX<Scalar>::Ones(n);
    TensorTrain<Scalar> speed = makeRank1(ones, ones, c_z);

    // Point-like source: a separable Gaussian ball (rank 1), centered in x, y
    // and near the top in z, two cells wide.
    const Scalar sigma = Scalar(2) * h;
    const std::vector<Scalar> source_at{extent / Scalar(2), extent / Scalar(2),
                                        extent / Scalar(5)};
    const auto gaussian = [&](std::size_t k) {
        Eigen::VectorX<Scalar> g(n);
        for (Eigen::Index i = 0; i < n; ++i) {
            const Scalar d = grid.coordinate(k, i) - source_at[k];
            g(i) = std::exp(-d * d / (Scalar(2) * sigma * sigma));
        }
        return g;
    };
    TensorTrain<Scalar> source_spatial =
        makeRank1(gaussian(0), gaussian(1), gaussian(2));

    // Ricker wavelet, f0 = 10 Hz as in the layered experiment of the report,
    // delayed so it starts from (numerical) zero.
    const Scalar f0 = Scalar(10); // Hz
    const Scalar t0 = Scalar(1.5) / f0;
    const auto ricker = [f0, t0](Scalar t) {
        const Scalar a = Scalar(M_PI) * f0 * (t - t0);
        const Scalar a2 = a * a;
        return (Scalar(1) - Scalar(2) * a2) * std::exp(-a2);
    };

    // CFL of the second-order stencil in 3D: dt < h / (c_max sqrt(3)); take
    // 1/2 of it.
    const Scalar dt = Scalar(0.5) * h / (c_bottom * std::sqrt(Scalar(3)));
    const int n_steps = static_cast<int>(std::ceil(t_end / dt));
    const int save_every = std::max(1, n_steps / 60);

    std::cout << "grid " << n << "^3, h = " << h << " m, dt = " << dt << " s, "
              << n_steps << " steps to t = " << t_end << " s\n"
              << "rtol = " << rtol << ", width policy = ceil("
              << width_factor << " * r) + " << oversampling << '\n';

    auto rhs = std::make_unique<AcousticRhs<Scalar>>(
        std::move(speed), source_spatial, ricker, sizes, spacings);

    // Cerjan sponge on all walls.
    const Eigen::Index sponge_width = 10;
    auto boundary = std::make_unique<MaskBoundaryCondition<Scalar>>(
        makeCerjanMask<Scalar>(sizes, sponge_width, Scalar(0.03)));

    // The wavefield starts at rest: p_{-1} = p_0 = 0 as rank-1 zero trains.
    // The first skeleton is then arbitrary, but its fibers are exactly zero,
    // so step 1 reduces to the source deposit alone.
    const auto rest_state = [&]() {
        return makeRank1(Eigen::VectorX<Scalar>::Zero(n),
                         Eigen::VectorX<Scalar>::Zero(n),
                         Eigen::VectorX<Scalar>::Zero(n));
    };
    std::vector<TensorTrain<Scalar>> initial_history;
    initial_history.push_back(rest_state()); // p_{-1}
    initial_history.push_back(rest_state()); // p_0

    auto selector = std::make_unique<
        MatSubset::ForwardIterativeVolumeSamplingSelector<Scalar>>();
    const Scalar atol = Scalar(0);

    Solver<Scalar> solver(std::move(initial_history), std::move(rhs),
                          Scheme<Scalar>::leapfrogSecondOrder(), dt,
                          std::move(selector), atol, rtol, num_samples,
                          std::move(boundary));

    std::vector<
        std::unique_ptr<MatSubset::Experiments::FieldMapperBase<Scalar>>>
        fields;
    fields.push_back(std::make_unique<IdentityFieldMapper<Scalar>>("pressure"));
    SnapshotSaver<Scalar> saver(
        grid, std::move(fields), "snapshots", "wavefield",
        MatSubset::Experiments::StoragePrecision::Float);

    saver.save(solver.getState(), solver.time());
    for (int step = 1; step <= n_steps; ++step) {
        solver.step();
        if (step % save_every == 0 || step == n_steps) {
            const std::string path =
                saver.save(solver.getState(), solver.time());
            std::cout << "step " << step << " / " << n_steps
                      << ", t = " << solver.time()
                      << " s, max rank = " << maxRank(solver.getState())
                      << ", wrote " << path << '\n';
        }
    }

    // Raw dump of the final field (Fortran/first-mode-fastest order) for
    // numerical comparison against a dense reference solve.
    {
        const Eigen::VectorXd dense = solver.getState().toDense();
        std::ofstream out("snapshots/final_field.bin", std::ios::binary);
        out.write(reinterpret_cast<const char *>(dense.data()),
                  static_cast<std::streamsize>(sizeof(double)) * dense.size());
    }

    return 0;
}
