// Accuracy comparison of column-subset-selection algorithms driving the TT
// cross skeleton of a low-rank integrator for the 3D acoustic wave equation
//   1/c^2 p_tt = lap(p) + s(x) f(t)   on the cube [0, extent]^3.
//
// Companion to the Allen-Cahn tester, and deliberately its opposite in the one
// respect that matters. Allen-Cahn is diffusive: it *smooths*, the state stays
// low-rank, and pinning every bond to a fixed rank is a fair way to compare
// selectors at equal cost. A wave has no such luxury — the wavefront sweeps
// outward and the rank needed to represent it grows with it. So this run is
// *tolerance-driven*, not fixed-rank: `rtol` decides every bond and the rank
// goes wherever the physics takes it.
//
// That changes what can be plotted. With the rank free, there is no single
// "best rank-r" floor to compare against — the floor would move under every
// curve, and differently for each selector. What takes its place is the rank
// itself: a selector that holds a given error at a lower rank is doing better
// work, and one that lets the rank run away is failing even if its error looks
// fine. Hence the second output column and the second panel of the figure.
//
// For each selector this runs a full integration with the `AdaptiveSolver`
// (which re-selects the skeleton against each stage combo, rather than
// carrying a fixed one), alongside a single dense full-grid reference
// (`DenseSolver`) running the identical discretization with no compression. At
// every snapshot it records the relative L2 error ||p - p_ref|| / ||p_ref||
// against that reference and the rank the state reached.
//
// Usage:
//   AcousticTester <path_to_config.json>
//
// The whole run is described by one JSON config, exactly as the Allen-Cahn
// tester's is. Algorithms are built by the shared `DefaultSelectorFactory`, so
// every selector the `Tester` accepts works here with the identical
// `{"name": ..., ...}` spelling and no changes to this file.
//
//   {
//     "output_path": "results/acoustic",     // relative to the config's parent
//     "n": 64,                               // grid points per axis
//     "rtol": 1e-5,                          // truncation tolerance (drives the rank)
//     "max_rank": 0,                         // 0 = uncapped; a safety net, not a target
//     "t_end": 1.0,                          // seconds
//     "extent": 2000.0,                      // metres per axis
//     "c_speed": 1500.0,                     // homogeneous medium speed, m/s
//     "f0": 10.0,                            // Ricker centre frequency, Hz
//     "order": 8,                            // Laplacian stencil order
//     "snapshots": 100,                      // error samples over the run
//     "width_factor": 1.0,                   // width = ceil(f * r) + oversampling
//     "oversampling": 2,                     // additive margin; see the note below
//     "warmup_time": 0.15,                   // exact TT arithmetic up to this t
//     "warmup_steps": 10,                    // ... or the same as a step count
//                                            //     ("warmup_time" wins if both)
//     "trials_per_algorithm": 16,            // seeds for randomized algorithms
//     "threads": 0,                          // 0 = hardware concurrency
//     "unfolding_time": 0.5,                 // when to dump the unfolding
//                                            //     (negative disables it)
//     "algorithms": [
//       {"display_name": "FDVS", "name": "derandomized volume"}
//     ]
//   }
//
// The medium is homogeneous for now (a single "c_speed"), so the speed train is
// the rank-1 constant field; the layered model of `main_acoustic.cpp` is a
// one-line change to `speedAxis` when that becomes the subject.
//
// The skeleton must be *wider* than the current rank for the rank to adapt: it
// is the spare columns that reveal the directions the state is about to need.
// With `"width_factor": 1.0` and no oversampling the rank freezes at its
// initial value (~2) and the error climbs to O(1) with the rank column
// perfectly flat, so the config parser rejects that combination outright.
//
// The width is deliberately set narrow (1.0 * r + 4), close to the point where
// the weaker selectors start to struggle, since a policy generous enough for
// every algorithm to succeed measures nothing. See the Config::width_factor
// comment for the sweep behind that choice — in particular that it is the
// *additive* margin that decides whether a bond can grow at all, and that
// getting it wrong silently misattributes a width-policy artefact to the
// selectors (Dominant finishes 1500x worse with no additive margin, despite
// selecting columns just as well as the winners).
//
// A consequence worth stating plainly: the interpolation-point baselines
// `"deim"` and `"qdeim"` return exactly one index per basis vector, i.e. they
// require width == rank, which is precisely the case the rank cannot adapt in.
// They are therefore *not available* in this experiment, unlike the fixed-rank
// Allen-Cahn one where they are the natural baselines. Oversampled Q-DEIM is a
// genuine algorithm and would be the way to bring that family here, but it is
// a different method and is not implemented.
//
// A warning about the randomized selectors on this problem: the wave state is
// far less forgiving than Allen-Cahn's, and a sampling-based skeleton that
// misses the wavefront produces a state whose error grows without bound. VS in
// particular is expected to diverge here, which is why the shipped config
// leaves it out. Diverging runs are not errors — they are recorded like any
// other and simply run off the top of the figure.
//
// `"randomized": true` marks an algorithm to be re-run under
// `trials_per_algorithm` distinct seeds (injected as the `"seed"` the factory
// reads); everything else runs once, since a deterministic selector returns the
// same skeleton every call.
//
// Runs are executed in parallel across cores: each (algorithm, seed) pair is an
// independent integration, so they are dispatched over a thread pool and their
// rows collected in deterministic order afterwards.
//
// Output (written to "output_path"):
//   acoustic_errors.csv  columns: algorithm,sample,step,t,error,max_rank,ranks
// plus, at "unfolding_time", a dump of one TT unfolding of the solver's state
// (acoustic_unfolding.csv) so the standard `Tester` pipeline can be pointed at
// a matrix this problem genuinely produces.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <pthread.h> // For the large-stack worker threads (see runOnBigStackThreads)

#include <Eigen/Core>
#include <Eigen/SVD> // For BDCSVD, to score the captured matrices

#include <nlohmann/json.hpp>

// The Tester's own selector factory: every algorithm name the standard
// experiment configs accept is registered there, so the roster of this
// experiment stays in sync with that one for free.
#include <SelectorFactory.h>

#include <AcousticEquation/AcousticRhs.h>
#include <AcousticEquation/DenseSolver.h>
#include <AcousticEquation/FiniteDifference.h>
#include <TTCrossSolver/AdaptiveSolver.h>
#include <TTCrossSolver/TensorTrain.h>
#include <TTCrossSolver/TensorTrainCore.h>

using Scalar = double;

using MatSubset::Experiments::AcousticRhs;
using MatSubset::Experiments::AdaptiveSolver;
using MatSubset::Experiments::DenseSolver;
using MatSubset::Experiments::Scheme;
using MatSubset::Experiments::secondDerivativeSpectralFactor;
using MatSubset::Experiments::TensorTrain;
using MatSubset::Experiments::TensorTrainCore;

namespace {

// ── the config ────────────────────────────────────────────────────────────────

// One algorithm of the roster, exactly as the config spells it. `config` is the
// verbatim JSON node, handed to the shared factory to build the selector — so
// algorithm-specific parameters ("c", "initialization", "eps", …) need no
// support here. `randomized` marks the algorithms worth re-running under
// several seeds.
struct AlgorithmSpec {
    std::string display_name;
    nlohmann::json config;
    bool randomized;
};

// Every knob of the experiment, parsed from the config file.
struct Config {
    Eigen::Index n = 64;              // grid points per axis
    // Truncation tolerance; drives the rank. 1e-5 rather than the 1e-6 of
    // main_acoustic.cpp: the tighter value is only needed to catch the rank
    // growth through the startup, and with the warm-up now covering that phase
    // in exact arithmetic it buys nothing but a smaller error and a larger
    // rank. Loosening it keeps the comparison in a regime where the selectors
    // are actually working for their accuracy.
    Scalar rtol = Scalar(1e-5);
    Eigen::Index max_rank = 0;        // 0 = uncapped
    Scalar t_end = Scalar(1);         // final time, seconds
    Scalar extent = Scalar(2000);     // metres per axis
    Scalar c_speed = Scalar(1500);    // homogeneous medium speed, m/s
    Scalar f0 = Scalar(10);           // Ricker centre frequency, Hz
    int order = 8;                    // Laplacian stencil order
    int snapshots = 100;              // error samples over the run
    // width = ceil(f * r) + oversampling. Strictly greater than 1 here, unlike
    // the fixed-rank Allen-Cahn run: a skeleton exactly as wide as the current
    // rank has no spare columns from which to discover that the rank should
    // grow, so the state freezes at its initial rank and the error runs away.
    //
    // The skeleton width is not a formality — it is the parameter that decides
    // how much room a selector has to work in, and the run is deliberately
    // placed at a *narrow* setting so the selectors are compared near their
    // limit rather than in the regime where every one of them succeeds.
    //
    // Measured over a sweep of (width_factor, oversampling) at n = 64, the two
    // knobs are not interchangeable. What governs whether a bond can grow is
    // the *additive* margin, and the transition in it is a cliff, not a slope:
    //
    //   oversampling <= 1   Dominant and Dominant-split collapse to ~0.29 at
    //                       every width_factor tried (1.0, 1.25, 1.5), a
    //                       1500-2000x spread across the roster
    //   oversampling >= 2   they are healthy at every width_factor tried,
    //                       and the spread falls to 2.5-4.6x
    //
    // The collapse is *not* a selection failure. Instrumenting the run shows
    // Dominant's column choices are as good as the winners' (median
    // ||X_S^+||/||X^+|| of 2.41 against Frobenius selection's 2.27); what
    // happens instead is that it starves one of the two bonds, holding it at
    // rank 2 through the steps where the wavefront first develops structure
    // (38 selection calls on a rank-2 bond over steps 36-54, against zero for
    // Frobenius selection). Leapfrog never recovers the error taken on there.
    //
    // So 1.0 * r + 2 is chosen as the narrowest policy that still leaves every
    // selector standing: it is on the safe side of the cliff, but only just.
    // It separates the roster in *both* panels rather than only the error one —
    // VS pays rank ~32 for its accuracy where every deterministic selector
    // holds ~17, which is exactly the trade the rank panel exists to show.
    // Widening to +4 compresses the spread and flattens the rank panel;
    // narrowing to +1 destroys Dominant outright.
    //
    // Note that +2 is only safe here in combination with the long warm-up (see
    // Config::warmup_time): with the short one it is on the *wrong* side of the
    // cliff and Dominant partially collapses. The two settings are not
    // independent — the warm-up decides how developed the state is when the
    // first skeleton is selected, and a state with more structure tolerates a
    // narrower skeleton.
    Scalar width_factor = Scalar(1);
    Eigen::Index oversampling = 2;
    // Exact TT steps before any skeleton is selected. Specified either as a
    // step count or, via "warmup_time", as a physical duration — the latter is
    // usually what is meant ("let the wave leave the source first") and does
    // not have to be recomputed when the grid, and with it dt, changes.
    //
    // 0.2 s is long enough for the wavefront to be well clear of the source and
    // carrying real structure before the first skeleton is chosen. That is
    // worth two things beyond realism. It removes the startup transient from
    // the measurement — the field no longer begins as exact zeros from a
    // low-rank selector's point of view, so the error starts at ~1e-5 instead
    // of at machine epsilon and the figure spans six decades rather than
    // fifteen. And a state that already has structure tolerates a narrower
    // skeleton, which is what makes the aggressive +2 width policy viable.
    int warmup_steps = 10;
    Scalar warmup_time = Scalar(0.2); // < 0: use warmup_steps instead
    int trials = 16;                  // seeds per randomized algorithm
    int threads = 0;                  // 0 = hardware concurrency
    // The time (in seconds, like t_end) at which to dump the unfolding matrix
    // the cross steps actually select from. 0.5 s is mid-run: the wavefront is
    // well developed and has just begun reflecting, so the state carries real
    // structure rather than the near-separable early field. Negative disables
    // the dump.
    Scalar unfolding_time = Scalar(0.5);
    std::filesystem::path output_path;
    std::vector<AlgorithmSpec> algorithms;
};

/*!
 * @brief Parses the experiment config.
 * @param path The config file.
 *
 * "output_path" is resolved relative to the config's own parent directory,
 * matching how the `Tester`'s configs behave, so a config and its results
 * travel together whatever directory the binary is run from.
 */
Config parseConfig(const std::filesystem::path &path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open the config: " + path.string());
    }
    const nlohmann::json j = nlohmann::json::parse(in);

    Config c;
    c.n = j.value("n", c.n);
    c.rtol = j.value("rtol", c.rtol);
    c.max_rank = j.value("max_rank", c.max_rank);
    c.t_end = j.value("t_end", c.t_end);
    c.extent = j.value("extent", c.extent);
    c.c_speed = j.value("c_speed", c.c_speed);
    c.f0 = j.value("f0", c.f0);
    c.order = j.value("order", c.order);
    c.snapshots = j.value("snapshots", c.snapshots);
    c.width_factor = j.value("width_factor", c.width_factor);
    c.oversampling = j.value("oversampling", c.oversampling);
    c.warmup_steps = j.value("warmup_steps", c.warmup_steps);
    c.warmup_time = j.value("warmup_time", c.warmup_time);
    c.trials = j.value("trials_per_algorithm", c.trials);
    c.threads = j.value("threads", c.threads);
    c.unfolding_time = j.value("unfolding_time", c.unfolding_time);

    const std::filesystem::path parent =
        path.has_parent_path() ? path.parent_path() : std::filesystem::path(".");
    c.output_path = parent / j.value("output_path", std::string("results"));

    for (const nlohmann::json &a : j.at("algorithms")) {
        // "display_name" is what the CSV and the legend carry; it falls back to
        // the algorithm name, exactly as the Tester does.
        const std::string name = a.at("name").get<std::string>();
        c.algorithms.push_back({a.value("display_name", name), a,
                                a.value("randomized", false)});
    }
    if (c.algorithms.empty()) {
        throw std::runtime_error("the config lists no algorithms");
    }
    if (c.rtol < Scalar(0)) {
        throw std::runtime_error("\"rtol\" must be >= 0");
    }
    // A skeleton exactly as wide as the current rank has no spare columns from
    // which to notice that the rank should grow, so on this problem the state
    // silently freezes at rank ~2 and the error runs away to O(1) while the
    // rank column stays flat — a failure that reads like a bad selector rather
    // than a bad config. Refuse it outright.
    //
    // The condition is on the width *at rank 1*, i.e. ceil(width_factor) +
    // oversampling, rather than on width_factor alone: what a bond needs in
    // order to grow is spare columns, and the additive margin supplies them
    // just as well as the multiplicative one. width_factor = 1 with
    // oversampling = 4 is a perfectly good (and deliberately narrow) policy;
    // width_factor = 1.5 with oversampling = 0 is not, since ceil(1.5 * 2) = 3
    // leaves a rank-2 bond a single spare column.
    const Eigen::Index width_at_rank_one =
        static_cast<Eigen::Index>(std::ceil(c.width_factor)) + c.oversampling;
    if (width_at_rank_one < 2) {
        throw std::runtime_error(
            "the rank cannot adapt with a skeleton no wider than the rank: "
            "raise \"width_factor\" or \"oversampling\" (this run is "
            "tolerance-driven, unlike the fixed-rank Allen-Cahn one). Note "
            "this rules out the \"deim\" and \"qdeim\" baselines, which "
            "require width == rank.");
    }
    return c;
}

// ── small helpers ─────────────────────────────────────────────────────────────

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
// trivial boundary ranks r_0 = r_d = 1.
std::string interiorRanks(const TensorTrain<Scalar> &train) {
    const std::vector<Eigen::Index> r = train.ranks();
    std::string out;
    for (std::size_t k = 1; k + 1 < r.size(); ++k) {
        out += (k > 1 ? "x" : "") + std::to_string(r[k]);
    }
    return out;
}

/*!
 * @brief A selector that records every matrix it is asked to select from, and
 * delegates the selection itself to a real selector.
 *
 * Used to capture the *actual* input of a cross step: what reaches a selector
 * is not a bare core unfolding but that unfolding contracted with the
 * accumulated factor from the cores already swept (`right_partial` /
 * `left_partial` in `TensorTrain::selectCrossIndices`, via
 * `CoreBase::absorbRight`/`absorbLeft`). Reconstructing that by hand is both
 * fiddly and easy to get subtly wrong, so instead the real sweep is run and
 * this listens in.
 */
class CapturingSelector : public MatSubset::SelectorBase<Scalar> {
  public:
    explicit CapturingSelector(
        std::unique_ptr<MatSubset::SelectorBase<Scalar>> inner)
        : inner(std::move(inner)) {}

    [[nodiscard]] std::string getAlgorithmName() const override {
        return "capturing(" + inner->getAlgorithmName() + ")";
    }

    /*! @brief Every matrix handed to the selector, in sweep order. */
    std::vector<Eigen::MatrixX<Scalar>> seen;

  protected:
    std::vector<Eigen::Index>
    selectSubsetImpl(const Eigen::MatrixX<Scalar> &X, Eigen::Index k,
                     Eigen::Index *swap_count) override {
        seen.push_back(X);
        (void)swap_count;
        return inner->selectSubset(X, k);
    }

  private:
    std::unique_ptr<MatSubset::SelectorBase<Scalar>> inner;
};

/*!
 * @brief Writes, as a plain CSV, the worst-conditioned matrix a cross step of
 * `train` actually hands its selector.
 *
 * Dumping a matrix here lets the standard `Tester` pipeline — and therefore the
 * two-panel figure of the superconductivity experiment — be pointed at a matrix
 * this PDE produces, instead of at a static dataset. *Which* matrix is dumped
 * matters a great deal.
 *
 * A core's bare unfolding is a degenerate test case. Each sweep of
 * `selectCrossIndices` orthogonalizes the core before selecting from it
 * (`rightSvd` in sweep 1, `leftOrth` in sweep 2), so the bare unfolding has
 * exactly orthonormal rows: every singular value 1, condition number 1. Every
 * selector's guarantee is trivially tight on such a matrix.
 *
 * What the selector is really given is that orthonormal core *contracted with
 * the accumulated factor of the cores already swept* (`absorbRight`/
 * `absorbLeft` of `right_partial`/`left_partial`). At a boundary core that
 * factor is still the 1x1 identity, so those captures are exactly orthonormal;
 * the interior ones are not, and are what this keeps.
 *
 * Unlike the Allen-Cahn tester, which dumps from a rank-truncated TT-SVD of the
 * dense reference, the train handed here is the *solver's own state*: with the
 * rank tolerance-driven rather than pinned there is no single rank at which to
 * re-truncate the reference, and the state the cross steps actually see is the
 * honest subject anyway.
 *
 * It is written transposed (tall, one row per column of the matrix) because
 * `MatrixFromFileGenerator` auto-transposes a tall matrix back to wide, which
 * is the orientation the selectors require.
 */
void writeUnfoldingCsv(const TensorTrain<Scalar> &train,
                       const MatSubset::Experiments::SelectorFactory<Scalar>
                           &factory,
                       const nlohmann::json &algorithm, Scalar atol,
                       Scalar rtol, const std::filesystem::path &path) {
    // Run a real cross selection on a copy of the train, listening in on every
    // matrix the selector is handed. No rank cap: this run is tolerance-driven,
    // so the sweep truncates exactly as the integration itself does.
    TensorTrain<Scalar> copy = train;
    auto capture =
        std::make_unique<CapturingSelector>(factory.create(algorithm));
    CapturingSelector *tap = capture.get();
    std::unique_ptr<MatSubset::SelectorBase<Scalar>> as_base(
        std::move(capture));
    copy.selectCrossIndices(as_base, atol, rtol, nullptr, 0);

    // Keep the captured matrix with the widest spectrum: the boundary cores
    // contribute a perfectly conditioned one (nothing has been absorbed yet),
    // so picking on conditioning is what selects an interior, informative one.
    const Eigen::MatrixX<Scalar> *best = nullptr;
    Scalar best_spread = Scalar(-1);
    for (const Eigen::MatrixX<Scalar> &X : tap->seen) {
        if (X.rows() < 2 || X.cols() < X.rows()) {
            continue;
        }
        Eigen::BDCSVD<Eigen::MatrixX<Scalar>> svd(X);
        const auto &sv = svd.singularValues();
        const Scalar smallest = sv(sv.size() - 1);
        const Scalar spread = (smallest > Scalar(0))
                                  ? sv(0) / smallest
                                  : std::numeric_limits<Scalar>::infinity();
        if (spread > best_spread) {
            best_spread = spread;
            best = &X;
        }
    }
    if (!best) {
        throw std::runtime_error("no usable matrix was captured for the dump");
    }
    std::cout << "   captured " << tap->seen.size()
              << " selector inputs; keeping a " << best->rows() << " x "
              << best->cols() << " one (cond ~ " << best_spread << ")\n";

    // Transposed so the file is tall and survives the generator's auto-transpose.
    const Eigen::MatrixX<Scalar> M = best->transpose();
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("cannot write " + path.string());
    }
    out.precision(17);
    for (Eigen::Index i = 0; i < M.rows(); ++i) {
        for (Eigen::Index j = 0; j < M.cols(); ++j) {
            out << (j ? "," : "") << M(i, j);
        }
        out << '\n';
    }
    std::cout << "wrote " << path << " (" << M.rows() << " x " << M.cols()
              << ")\n";
}

/*!
 * @brief Runs `body` on `count` worker threads, each with a large stack, and
 * joins them.
 *
 * Not `std::thread`: on musl (Alpine) a spawned thread gets a ~128 KB stack,
 * where the process's main thread gets 8 MB. Eigen's decompositions put
 * sizeable temporaries on the stack, so work that is fine on the main thread
 * segfaults inside a default-stack worker as soon as the grid grows. pthreads
 * is the only portable way to set the stack size, so the pool is built directly
 * on it.
 *
 * `body` must not throw: it runs as a C callback, and an exception escaping it
 * would terminate. The call site keeps its work inside a try/catch that stores
 * the message instead.
 */
void runOnBigStackThreads(unsigned int count,
                          const std::function<void()> &body) {
    // 64 MB: comfortably above what the decompositions here need, and it is
    // address space, not resident memory.
    constexpr std::size_t stack_bytes = std::size_t(64) << 20;

    pthread_attr_t attr;
    if (pthread_attr_init(&attr) != 0) {
        throw std::runtime_error("pthread_attr_init failed");
    }
    pthread_attr_setstacksize(&attr, stack_bytes);

    // The callback receives the same `body` for every worker; each decides
    // what to do by pulling from the shared atomic cursor its closure holds.
    const auto trampoline = [](void *arg) -> void * {
        (*static_cast<const std::function<void()> *>(arg))();
        return nullptr;
    };

    std::vector<pthread_t> threads;
    threads.reserve(count);
    for (unsigned int w = 0; w < count; ++w) {
        pthread_t tid;
        if (pthread_create(&tid, &attr, trampoline,
                           const_cast<std::function<void()> *>(&body)) != 0) {
            break; // Out of threads: the ones already running still finish.
        }
        threads.push_back(tid);
    }
    pthread_attr_destroy(&attr);

    if (threads.empty()) {
        body(); // Could not spawn anything; run it inline rather than skip.
    }
    for (const pthread_t tid : threads) {
        pthread_join(tid, nullptr);
    }
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_config.json>\n";
        return 1;
    }

    Config opt;
    try {
        opt = parseConfig(argv[1]);
    } catch (const std::exception &e) {
        std::cerr << "Failed to read the config: " << e.what() << '\n';
        return 1;
    }

    // One factory for the whole run: `create` is const and touches no shared
    // state, so the workers can call it concurrently.
    const MatSubset::Experiments::DefaultSelectorFactory<Scalar> factory;

    // Fail fast on an unknown algorithm name or a bad parameter, before the
    // dense reference trajectory is computed.
    for (const AlgorithmSpec &spec : opt.algorithms) {
        try {
            factory.create(spec.config);
        } catch (const std::exception &e) {
            std::cerr << "Bad algorithm \"" << spec.display_name
                      << "\": " << e.what() << '\n';
            return 1;
        }
    }

    // Tolerance-driven run: `rtol` decides every bond, so the rank is free to
    // follow the wavefront. `atol` stays at zero so the tail rule is purely
    // relative, and `max_rank` (0 by default) is only a safety net.
    const Scalar atol = Scalar(0);
    const Scalar rtol = opt.rtol;

    const Scalar extent = opt.extent;
    const Scalar h = extent / static_cast<Scalar>(opt.n - 1); // m

    const std::vector<Eigen::Index> sizes{opt.n, opt.n, opt.n};
    const std::vector<Scalar> spacings{h, h, h};

    // Homogeneous medium: c is the same everywhere, so the speed train is the
    // rank-1 constant field. (The layered model of main_acoustic.cpp replaces
    // the z axis here.)
    const Eigen::VectorX<Scalar> ones = Eigen::VectorX<Scalar>::Ones(opt.n);
    const Eigen::VectorX<Scalar> c_z =
        Eigen::VectorX<Scalar>::Constant(opt.n, opt.c_speed);

    // Source: a separable Gaussian ball (rank 1), centred in x and y and near
    // the top in z, two cells wide.
    const Scalar sigma = Scalar(2) * h;
    const std::vector<Scalar> source_at{extent / Scalar(2), extent / Scalar(2),
                                        extent / Scalar(5)};
    // The axis coordinate is just i * h on the uniform grid [0, extent], the
    // same value Grid::coordinate returns — computed directly so this file
    // needs no Grid (and so none of the VTK-adjacent headers that define it).
    const auto sourceAxis = [&](std::size_t k) {
        Eigen::VectorX<Scalar> v(opt.n);
        for (Eigen::Index i = 0; i < opt.n; ++i) {
            const Scalar d = static_cast<Scalar>(i) * h - source_at[k];
            v(i) = std::exp(-d * d / (Scalar(2) * sigma * sigma));
        }
        return v;
    };
    const Eigen::VectorX<Scalar> src0 = sourceAxis(0);
    const Eigen::VectorX<Scalar> src1 = sourceAxis(1);
    const Eigen::VectorX<Scalar> src2 = sourceAxis(2);

    // Ricker wavelet, delayed so it starts from (numerical) zero.
    const Scalar f0 = opt.f0;
    const Scalar t0 = Scalar(1.5) / f0;
    const auto ricker = [f0, t0](Scalar t) {
        const Scalar a = Scalar(M_PI) * f0 * (t - t0);
        const Scalar a2 = a * a;
        return (Scalar(1) - Scalar(2) * a2) * std::exp(-a2);
    };

    // CFL for leapfrog on the order-`order` Laplacian in d = 3 dimensions:
    // stability needs dt <= 2 / sqrt(lambda_max) with the Laplacian spectral
    // radius lambda_max = d * S / h^2, S the per-axis spectral factor of the
    // stencil. We take 1/2 of the limit, as main_acoustic.cpp does.
    const Scalar spectral_factor =
        secondDerivativeSpectralFactor<Scalar>(opt.order);
    const Scalar dt =
        Scalar(0.5) * Scalar(2) * h /
        (opt.c_speed * std::sqrt(Scalar(sizes.size()) * spectral_factor));
    const int n_steps = static_cast<int>(std::ceil(opt.t_end / dt));
    const int save_every = std::max(1, n_steps / std::max(1, opt.snapshots));

    // "warmup_time", when given, wins over "warmup_steps": the duration is the
    // physically meaningful quantity (how long the wave is allowed to develop
    // before any skeleton is selected), and expressing it in steps would have
    // to be redone for every grid, since dt scales with h.
    const int warmup_steps =
        (opt.warmup_time >= Scalar(0))
            ? static_cast<int>(std::ceil(opt.warmup_time / dt))
            : opt.warmup_steps;
    if (warmup_steps >= n_steps) {
        std::cerr << "the warm-up covers the whole run (" << warmup_steps
                  << " of " << n_steps
                  << " steps): nothing would be measured. Lower "
                     "\"warmup_time\" or raise \"t_end\".\n";
        return 1;
    }

    const Scheme<Scalar> scheme = Scheme<Scalar>::leapfrogSecondOrder();

    const auto num_samples = [&opt](Eigen::Index rank, Eigen::Index) {
        return static_cast<Eigen::Index>(
                   std::ceil(opt.width_factor * static_cast<Scalar>(rank))) +
               opt.oversampling;
    };

    const unsigned int n_threads =
        (opt.threads > 0)
            ? static_cast<unsigned int>(opt.threads)
            : std::max(1u, std::thread::hardware_concurrency());

    std::cout << "grid " << opt.n << "^3, h = " << h << " m, dt = " << dt
              << " s, " << n_steps << " steps to t = " << opt.t_end << " s\n"
              << "c = " << opt.c_speed << " m/s (homogeneous), f0 = " << f0
              << " Hz, rtol = " << rtol << " (atol = 0, rank free)"
              << ", leapfrog, stencil order = " << opt.order
              << ", width policy = ceil(" << opt.width_factor << " * r) + "
              << opt.oversampling << ", warmup = " << warmup_steps
              << " steps (t = " << static_cast<Scalar>(warmup_steps) * dt << ")"
              << ", trials = " << opt.trials << ", threads = " << n_threads
              << '\n';

    const std::filesystem::path &out_dir = opt.output_path;
    std::filesystem::create_directories(out_dir);

    // ── the dense reference trajectory ────────────────────────────────────────
    // Run once, up front, and keep the field at each snapshot step. Every
    // selector's run is then scored against the same stored trajectory instead
    // of re-integrating the (expensive) full-grid solver once per algorithm.
    std::cout << "\n==> dense reference trajectory\n";
    std::vector<Eigen::VectorX<Scalar>> reference_fields;
    std::vector<int> reference_steps;
    std::vector<Scalar> reference_times;
    {
        const Eigen::Index total = opt.n * opt.n * opt.n;
        std::vector<Eigen::VectorX<Scalar>> initial_history;
        initial_history.push_back(Eigen::VectorX<Scalar>::Zero(total)); // p_{-1}
        initial_history.push_back(Eigen::VectorX<Scalar>::Zero(total)); // p_0

        DenseSolver<Scalar> dense(denseRank1(ones, ones, c_z),
                                  denseRank1(src0, src1, src2), ricker, sizes,
                                  spacings, scheme, dt,
                                  std::move(initial_history), opt.order);
        for (int step = 1; step <= n_steps; ++step) {
            dense.step();
            if (step % save_every == 0 || step == n_steps) {
                reference_fields.push_back(dense.field());
                reference_steps.push_back(step);
                reference_times.push_back(dense.time());
            }
        }
        std::cout << "   " << reference_fields.size() << " snapshots stored\n";
    }

    // ── the unfolding dump ────────────────────────────────────────────────────
    // Integrate once up to "unfolding_time" and dump the matrix a cross step of
    // that state actually hands its selector, so the standard `Tester` pipeline
    // — and with it the two-panel figure of the superconductivity experiment —
    // can be pointed at a matrix this wave problem genuinely produces.
    //
    // The state is taken from the solver rather than from a TT-SVD of the dense
    // reference (as the Allen-Cahn tester does): with the rank tolerance-driven
    // there is no single rank at which to re-truncate the reference, and the
    // train the cross steps actually see is the honest subject anyway.
    if (opt.unfolding_time >= Scalar(0)) {
        const int unfolding_step = std::min(
            n_steps,
            std::max(1, static_cast<int>(std::ceil(opt.unfolding_time / dt))));
        std::cout << "\n==> unfolding dump at t = "
                  << static_cast<Scalar>(unfolding_step) * dt << " (step "
                  << unfolding_step << " of " << n_steps << ")\n";

        std::string failure;
        // On a big stack for the same reason the integrations are: the capture
        // scores its matrices with BDCSVD.
        runOnBigStackThreads(1, [&] {
            try {
                const Eigen::VectorX<Scalar> zero =
                    Eigen::VectorX<Scalar>::Zero(opt.n);
                std::vector<TensorTrain<Scalar>> initial_history;
                initial_history.push_back(makeRank1(zero, zero, zero));
                initial_history.push_back(makeRank1(zero, zero, zero));

                AdaptiveSolver<Scalar> solver(
                    std::move(initial_history),
                    std::make_unique<AcousticRhs<Scalar>>(
                        makeRank1(ones, ones, c_z), makeRank1(src0, src1, src2),
                        ricker, sizes, spacings, opt.order),
                    scheme, dt, factory.create(opt.algorithms.front().config),
                    atol, rtol, num_samples, /*boundary_condition=*/nullptr,
                    warmup_steps);

                for (int step = 1; step <= unfolding_step; ++step) {
                    solver.step();
                }
                std::cout << "   state TT ranks "
                          << interiorRanks(solver.getState()) << '\n';

                // Which algorithm drives the capturing sweep only affects which
                // columns that sweep keeps as it moves along the train, not the
                // property being dumped (the absorbed, non-orthonormal
                // unfolding), so the first configured one stands in.
                writeUnfoldingCsv(solver.getState(), factory,
                                  opt.algorithms.front().config, atol, rtol,
                                  out_dir / "acoustic_unfolding.csv");
            } catch (const std::exception &e) {
                failure = e.what();
            }
        });
        if (!failure.empty()) {
            std::cerr << "The unfolding dump failed: " << failure << '\n';
            return 1;
        }
    }

    // ── per-selector runs, in parallel ────────────────────────────────────────
    // Each (selector, seed) pair is a completely independent integration —
    // nothing is shared but the read-only reference trajectory — so the pairs
    // are flattened into one job list and handed to a thread pool. Each job
    // writes its rows into its own slot of `rows`, which is then emitted in job
    // order: the CSV is byte-identical whatever order the threads finish in.
    struct Job {
        std::string display_name;
        int sample;
        nlohmann::json config; // the algorithm node, with "seed" filled in
    };
    std::vector<Job> jobs;
    for (const AlgorithmSpec &spec : opt.algorithms) {
        const int runs = spec.randomized ? opt.trials : 1;
        for (int sample = 0; sample < runs; ++sample) {
            // The seed is injected into the algorithm's own node, which is
            // where the factory looks for it. A config that pins its own
            // "seed" keeps it, so a single-seed rerun stays reproducible.
            nlohmann::json algorithm = spec.config;
            if (spec.randomized && !algorithm.contains("seed")) {
                algorithm["seed"] =
                    static_cast<std::mt19937::result_type>(sample);
            }
            jobs.push_back({spec.display_name, sample, std::move(algorithm)});
        }
    }

    std::cout << "\n==> " << jobs.size() << " integrations on " << n_threads
              << " threads\n";

    std::vector<std::string> rows(jobs.size());
    {
        std::atomic<std::size_t> next{0};
        std::mutex log_mutex;
        std::string failure;
        const unsigned int workers = std::min<unsigned int>(
            n_threads,
            static_cast<unsigned int>(std::max<std::size_t>(jobs.size(), 1)));
        runOnBigStackThreads(workers, [&] {
            try {
                for (std::size_t j = next++; j < jobs.size(); j = next++) {
                    const Job &job = jobs[j];

                    // The wavefield starts at rest as exact zero states. The
                    // first steps run in exact TT arithmetic (the warm-up), so
                    // no skeleton is ever selected from a structureless state.
                    const Eigen::VectorX<Scalar> zero =
                        Eigen::VectorX<Scalar>::Zero(opt.n);
                    std::vector<TensorTrain<Scalar>> initial_history;
                    initial_history.push_back(makeRank1(zero, zero, zero));
                    initial_history.push_back(makeRank1(zero, zero, zero));

                    // The adaptive solver re-selects the skeleton against each
                    // stage combo instead of carrying a fixed one, which is
                    // what lets the rank track the wavefront.
                    AdaptiveSolver<Scalar> solver(
                        std::move(initial_history),
                        std::make_unique<AcousticRhs<Scalar>>(
                            makeRank1(ones, ones, c_z),
                            makeRank1(src0, src1, src2), ricker, sizes,
                            spacings, opt.order),
                        scheme, dt, factory.create(job.config), atol, rtol,
                        num_samples, /*boundary_condition=*/nullptr,
                        warmup_steps);

                    std::ostringstream out;
                    out.precision(17);

                    std::size_t snapshot = 0;
                    for (int step = 1; step <= n_steps; ++step) {
                        solver.step();
                        if (snapshot >= reference_steps.size() ||
                            step != reference_steps[snapshot]) {
                            continue;
                        }

                        const TensorTrain<Scalar> &state = solver.getState();
                        const Eigen::VectorX<Scalar> &ref =
                            reference_fields[snapshot];
                        const Scalar ref_norm = ref.norm();
                        const Scalar diff_norm = (state.toDense() - ref).norm();
                        const Scalar error = (ref_norm > Scalar(0))
                                                 ? diff_norm / ref_norm
                                                 : diff_norm;

                        // The rank profile is 'x'-separated, so it is quoted:
                        // it is the last column today, but a quoted field
                        // survives a reordering.
                        out << job.display_name << ',' << job.sample << ','
                            << step << ',' << reference_times[snapshot] << ','
                            << error << ',' << maxRank(state) << ",\""
                            << interiorRanks(state) << "\"\n";
                        ++snapshot;
                    }
                    rows[j] = std::move(out).str();

                    const std::lock_guard<std::mutex> lock(log_mutex);
                    std::cout << "   " << job.display_name << " sample "
                              << job.sample << ": final ranks "
                              << interiorRanks(solver.getState()) << '\n';
                }
            } catch (const std::exception &e) {
                const std::lock_guard<std::mutex> lock(log_mutex);
                if (failure.empty()) {
                    failure = e.what();
                }
            }
        });
        if (!failure.empty()) {
            std::cerr << "An integration failed: " << failure << '\n';
            return 1;
        }
    }

    std::ofstream csv(out_dir / "acoustic_errors.csv");
    if (!csv) {
        throw std::runtime_error("cannot write the errors CSV");
    }
    csv << "algorithm,sample,step,t,error,max_rank,ranks\n";
    for (const std::string &row : rows) {
        csv << row;
    }

    std::cout << "\nwrote " << (out_dir / "acoustic_errors.csv") << '\n';
    return 0;
}
