// QTT (quantized TT) variant of the TT-leapfrog acoustic solve in main.cpp.
// Same physics — 1/c^2 p_tt = lap(p) + s(x) f(t) on an n^3 Dirichlet grid — but
// every axis of size n = 2^L is bit-split into L size-2 modes (grouped by axis,
// big-endian), so the state, the operator and the ICs live in 3L QTT cores. The
// solver, selector and adaptive machinery are mode-agnostic and unchanged; only
// the ICs, the RHS (QttAcousticRhs) and the field mapper (QttFieldMapper)
// differ from the TT run. Snapshots go to ./out/snapshots_qtt as .vti files.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include <Eigen/Core>

#include <MatSubset/FrobeniusSelectionSelector.h>

#include <AcousticEquation/AcousticRhs.h>     // for hadamardProduct via train
#include <AcousticEquation/QttAcousticRhs.h>  // QttAcousticRhs, trainToQtt, ...
#include <TTCrossSolver/AdaptiveSolver.h>
#include <TTCrossSolver/SnapshotSaver.h>
#include <TTCrossSolver/TensorTrain.h>
#include <TTCrossSolver/TensorTrainCore.h>

using Scalar = double;

using MatSubset::Experiments::AdaptiveSolver;
using MatSubset::Experiments::Grid;
using MatSubset::Experiments::QttAcousticRhs;
using MatSubset::Experiments::QttFieldMapper;
using MatSubset::Experiments::QttLayout;
using MatSubset::Experiments::Scheme;
using MatSubset::Experiments::SnapshotSaver;
using MatSubset::Experiments::TensorTrain;
using MatSubset::Experiments::TensorTrainCore;
using MatSubset::Experiments::trainToQtt;
using MatSubset::Experiments::vectorToQtt;

namespace {

// A rank-1 physical 3D train v0 x v1 x v2 (one core per axis), used only as the
// pre-QTT description of a separable field; trainToQtt bit-splits it.
TensorTrain<Scalar> makeRank1Physical(const Eigen::VectorX<Scalar> &v0,
                                      const Eigen::VectorX<Scalar> &v1,
                                      const Eigen::VectorX<Scalar> &v2) {
    std::vector<TensorTrainCore<Scalar>> cores;
    cores.emplace_back(Eigen::MatrixX<Scalar>(v0), v0.size());
    cores.emplace_back(Eigen::MatrixX<Scalar>(v1), v1.size());
    cores.emplace_back(Eigen::MatrixX<Scalar>(v2), v2.size());
    return TensorTrain<Scalar>(std::move(cores));
}

// An exact-zero QTT rest state: 3L size-2 cores, all zero. Built by bit-
// splitting a zero vector per axis and concatenating (all cores rank 1).
TensorTrain<Scalar> makeZeroQtt(Eigen::Index n) {
    const Eigen::VectorX<Scalar> zero = Eigen::VectorX<Scalar>::Zero(n);
    std::vector<TensorTrainCore<Scalar>> cores;
    for (int axis = 0; axis < 3; ++axis) {
        TensorTrain<Scalar> axis_qtt = vectorToQtt<Scalar>(zero);
        for (std::size_t level = 0; level < axis_qtt.modeSizes().size();
             ++level) {
            cores.push_back(axis_qtt.core(level));
        }
    }
    return TensorTrain<Scalar>(std::move(cores));
}

Eigen::Index maxRank(const TensorTrain<Scalar> &train) {
    Eigen::Index r = 0;
    for (const Eigen::Index rank : train.ranks()) {
        r = std::max(r, rank);
    }
    return r;
}

// The interior bond ranks as a compact "[r1 r2 ...]" string. train.ranks()
// includes the two boundary 1s (leftmost and rightmost), which carry no
// information, so they are dropped.
std::string rankList(const TensorTrain<Scalar> &train) {
    const std::vector<Eigen::Index> ranks = train.ranks();
    std::ostringstream os;
    os << '[';
    for (std::size_t k = 1; k + 1 < ranks.size(); ++k) {
        os << ranks[k];
        if (k + 2 < ranks.size()) {
            os << ' ';
        }
    }
    os << ']';
    return os.str();
}

} // namespace

int main(int argc, char **argv) {
    // Grid: n^3 points on the cube [0, extent]^3, z = depth. n MUST be a power
    // of two for the QTT layout (asserted downstream); default 128 = 2^7.
    const Eigen::Index n = (argc > 1) ? std::atoi(argv[1]) : 128;
    const Scalar t_end = (argc > 2) ? std::atof(argv[2]) : Scalar(1);
    const Scalar rtol = (argc > 3) ? std::atof(argv[3]) : Scalar(1e-6);
    const Scalar width_factor = (argc > 4) ? std::atof(argv[4]) : Scalar(1);
    const Eigen::Index oversampling = (argc > 5) ? std::atoi(argv[5]) : 12;

    if ((n & (n - 1)) != 0 || n < 2) {
        std::cerr << "QTT mode needs n to be a power of two (got " << n
                  << ").\n";
        return 1;
    }

    // QTT core layout: arg 7 is "grouped" (default) or "interleaved". The
    // interleaved layout groups the bit modes by scale, which tends to lower the
    // state's ranks for the (roughly isotropic) expanding wavefront.
    QttLayout layout = QttLayout::Grouped;
    const std::string layout_arg = (argc > 7) ? argv[7] : "grouped";
    if (layout_arg == "interleaved" || layout_arg == "i") {
        layout = QttLayout::Interleaved;
    } else if (layout_arg != "grouped" && layout_arg != "g") {
        std::cerr << "unknown layout '" << layout_arg
                  << "' (expected 'grouped' or 'interleaved').\n";
        return 1;
    }
    const bool interleaved = (layout == QttLayout::Interleaved);

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

    // 3-layer medium: c depends on depth only, so the physical speed is rank 1
    // (ones x ones x c_z); trainToQtt bit-splits it into a QTT.
    const Scalar c_top = Scalar(2000);    // m/s
    const Scalar c_middle = Scalar(2000); // m/s
    const Scalar c_bottom = Scalar(2000); // m/s
    Eigen::VectorX<Scalar> c_z(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        const Scalar z = grid.coordinate(2, i);
        c_z(i) = (z < extent / Scalar(3))               ? c_top
                 : (z < Scalar(2) * extent / Scalar(3)) ? c_middle
                                                        : c_bottom;
    }
    const Eigen::VectorX<Scalar> ones = Eigen::VectorX<Scalar>::Ones(n);
    TensorTrain<Scalar> speed_qtt =
        trainToQtt<Scalar>(makeRank1Physical(ones, ones, c_z), layout);

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
    TensorTrain<Scalar> source_qtt = trainToQtt<Scalar>(
        makeRank1Physical(gaussian(0), gaussian(1), gaussian(2)), layout);

    // Ricker wavelet, f0 = 10 Hz, delayed so it starts from (numerical) zero.
    const Scalar f0 = Scalar(10); // Hz
    const Scalar t0 = Scalar(1.5) / f0;
    const auto ricker = [f0, t0](Scalar t) {
        const Scalar a = Scalar(M_PI) * f0 * (t - t0);
        const Scalar a2 = a * a;
        return (Scalar(1) - Scalar(2) * a2) * std::exp(-a2);
    };

    // CFL of the second-order stencil in 3D: dt < h / (c_max sqrt(3)); half it.
    const Scalar dt = Scalar(0.5) * h / (c_bottom * std::sqrt(Scalar(3)));
    const int n_steps = static_cast<int>(std::ceil(t_end / dt));
    const int save_every = 10;

    std::cout << "QTT grid " << n << "^3 (" << 3 * static_cast<int>(std::log2(n))
              << " bit cores, " << (interleaved ? "interleaved" : "grouped")
              << " layout), h = " << h << " m, dt = " << dt << " s, " << n_steps
              << " steps to t = " << t_end << " s\n"
              << "rtol = " << rtol << ", width policy = ceil(" << width_factor
              << " * r) + " << oversampling << '\n';

    auto rhs = std::make_unique<QttAcousticRhs<Scalar>>(
        std::move(speed_qtt), source_qtt, ricker, sizes, spacings, layout);

    // The wavefield starts at rest as exact-zero QTT trains (p_{-1}, p_0).
    std::vector<TensorTrain<Scalar>> initial_history;
    initial_history.push_back(makeZeroQtt(n)); // p_{-1}
    initial_history.push_back(makeZeroQtt(n)); // p_0

    auto selector =
        std::make_unique<MatSubset::FrobeniusSelectionSelector<Scalar>>();
    const Scalar atol = Scalar(0);

    const int warmup_steps = (argc > 6) ? std::atoi(argv[6]) : 10;
    AdaptiveSolver<Scalar> solver(std::move(initial_history), std::move(rhs),
                                  Scheme<Scalar>::leapfrogSecondOrder(), dt,
                                  std::move(selector), atol, rtol, num_samples,
                                  /*boundary_condition=*/nullptr, warmup_steps);

    // The display grid is the physical grid; QttFieldMapper translates each
    // physical node into its 3L bit modes when reading the QTT state.
    std::vector<
        std::unique_ptr<MatSubset::Experiments::FieldMapperBase<Scalar>>>
        fields;
    fields.push_back(std::make_unique<QttFieldMapper<Scalar>>("pressure", sizes,
                                                              interleaved));
    // Separate output dirs per layout so grouped/interleaved runs don't clash.
    const std::string out_dir =
        interleaved ? "out/snapshots_qtt_interleaved" : "out/snapshots_qtt";
    SnapshotSaver<Scalar> saver(
        grid, std::move(fields), out_dir, "wavefield",
        MatSubset::Experiments::StoragePrecision::Float);

    saver.save(solver.getState(), solver.time());
    const auto solve_start = std::chrono::steady_clock::now();
    for (int step = 1; step <= n_steps; ++step) {
        solver.step();

        if (step % save_every == 0 || step == n_steps) {
            saver.save(solver.getState(), solver.time());
            const auto elapsed =
                std::chrono::duration<Scalar>(std::chrono::steady_clock::now() -
                                              solve_start)
                    .count();
            std::cout << "step " << step << " / " << n_steps
                      << ", t = " << solver.time()
                      << " s, max rank = " << maxRank(solver.getState())
                      << ", ranks = " << rankList(solver.getState())
                      << ", elapsed = " << elapsed << " s (" << elapsed / step
                      << " s/step)\n";
        }
    }
    const auto solve_seconds =
        std::chrono::duration<Scalar>(std::chrono::steady_clock::now() -
                                      solve_start)
            .count();
    std::cout << "QTT solve took " << solve_seconds << " s for " << n_steps
              << " steps (" << solve_seconds / n_steps << " s/step)\n";

    return 0;
}
