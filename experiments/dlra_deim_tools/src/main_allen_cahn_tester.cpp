// Accuracy comparison of column-subset-selection algorithms driving the TT
// cross skeleton of a low-rank integrator for the 3D Allen-Cahn equation
//   df/dt = kappa * lap(f) + f - f^3   on the periodic cube [0, 2*pi]^3.
//
// Companion to the superconductivity experiment: the selectors are compared on
// the matrices a *time-dependent* problem actually produces — the TT unfoldings
// of the evolving state — rather than on a fixed data matrix.
//
// The run is *fixed-rank*: atol = rtol = 0 disables tolerance-driven
// truncation entirely and `--rank` pins every bond, so no rank adaptivity
// takes place and all selectors are compared at exactly equal cost. The state
// sits at that rank from the initial condition to the final step.
//
// For each selector this runs a full integration with the fixed-skeleton
// `Solver` (not the adaptive one) under SSP-RK2, alongside a single dense
// full-grid reference (`AllenCahnDenseSolver`) that runs the identical
// discretization with no compression. At every snapshot it records
//   - the relative Frobenius error ||f - f_ref||_F / ||f_ref||_F of the TT
//     state against that dense reference, and
//   - the same error for the *best* TT approximation of the reference at that
//     same fixed rank (rank-truncated TT-SVD of the dense field).
// The latter is the error floor no rank-limited integrator can beat, so the gap
// between a selector's curve and it is the price of cross selection over the
// optimal projection.
//
// Usage:
//   AllenCahnTester <path_to_config.json>
//
// The whole run is described by one JSON config, in the same spirit as the
// `Tester`'s: the grid, rank, final time and the algorithm roster all live in
// the file, so a new comparison is a new config rather than a new set of
// command-line flags. Algorithms are built by the shared
// `DefaultSelectorFactory`, so every selector the `Tester` accepts — and any
// future one — works here with the identical `{"name": ..., ...}` spelling and
// no changes to this file.
//
//   {
//     "output_path": "results/allen_cahn",   // relative to the config's parent
//     "n": 32,                               // grid points per axis
//     "rank": 10,                            // fixed TT rank at every bond
//     "t_end": 5.0,
//     "kappa": 0.1,
//     "order": 4,                            // Laplacian stencil order
//     "snapshots": 40,                       // error samples over the run
//     "width_factor": 1.0,                   // width = ceil(f * r) + oversampling
//     "oversampling": 0,
//     "trials_per_algorithm": 16,            // seeds for randomized algorithms
//     "threads": 0,                          // 0 = hardware concurrency
//     "algorithms": [
//       {"display_name": "FDVS", "name": "derandomized volume"},
//       {"display_name": "VS",   "name": "forward iterative volume sampling",
//        "randomized": true}
//     ]
//   }
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
//   allen_cahn_errors.csv  columns: algorithm,sample,step,t,error,best_error,
//                                   max_rank,ranks
// plus, at the midpoint of the run, a dump of one TT unfolding of the reference
// state (allen_cahn_unfolding.csv) so the standard `Tester` pipeline can be
// pointed at a matrix this problem genuinely produces.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
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

#include <nlohmann/json.hpp>

// The Tester's own selector factory: every algorithm name the standard
// experiment configs accept is registered there, so the roster of this
// experiment stays in sync with that one for free.
#include <SelectorFactory.h>

#include <AllenCahnEquation/AllenCahnDenseSolver.h>
#include <AllenCahnEquation/AllenCahnInitialCondition.h>
#include <AllenCahnEquation/AllenCahnRhs.h>
#include <TTCrossSolver/SnapshotSaver.h> // For Grid (no snapshots are written)
#include <TTCrossSolver/Solver.h>
#include <TTCrossSolver/TensorTrain.h>

using Scalar = double;

using MatSubset::Experiments::AllenCahnDenseSolver;
using MatSubset::Experiments::AllenCahnRhs;
using MatSubset::Experiments::allenCahnInitial;
using MatSubset::Experiments::denseFieldFromFunction;
using MatSubset::Experiments::Grid;
using MatSubset::Experiments::Scheme;
using MatSubset::Experiments::secondDerivativeSpectralFactor;
using MatSubset::Experiments::Solver;
using MatSubset::Experiments::TensorTrain;
using MatSubset::Experiments::ttFromDenseTensor;

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

// Every knob of the experiment, parsed from the config file. The defaults are
// the values the volume-sampling figure was specified on, so a minimal config
// need only name its algorithms.
struct Config {
    Eigen::Index n = 32;             // grid points per axis
    Eigen::Index rank = 10;          // fixed TT rank at every bond
    Scalar t_end = Scalar(5);        // final time
    Scalar kappa = Scalar(0.1);
    int order = 4;                   // Laplacian stencil order
    int snapshots = 40;              // error samples over the run
    Scalar width_factor = Scalar(1); // width = ceil(f * r) + oversampling
    Eigen::Index oversampling = 0;
    int trials = 16;                 // seeds per randomized algorithm
    int threads = 0;                 // 0 = hardware concurrency
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
    c.rank = j.value("rank", c.rank);
    c.t_end = j.value("t_end", c.t_end);
    c.kappa = j.value("kappa", c.kappa);
    c.order = j.value("order", c.order);
    c.snapshots = j.value("snapshots", c.snapshots);
    c.width_factor = j.value("width_factor", c.width_factor);
    c.oversampling = j.value("oversampling", c.oversampling);
    c.trials = j.value("trials_per_algorithm", c.trials);
    c.threads = j.value("threads", c.threads);

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
    if (c.rank < 1) {
        throw std::runtime_error("\"rank\" must be >= 1");
    }
    return c;
}

// ── small helpers ─────────────────────────────────────────────────────────────

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
 * @brief The relative Frobenius error
 * \f$ \lVert f - f_{\mathrm{ref}} \rVert_F / \lVert f_{\mathrm{ref}} \rVert_F \f$.
 *
 * Both fields arrive as the flattened dense tensor, and the Frobenius norm of a
 * tensor is by definition the Euclidean norm of its entries, so `.norm()` on
 * the flattening *is* the Frobenius norm — no reshaping is needed.
 */
Scalar relativeFrobeniusError(const Eigen::VectorX<Scalar> &candidate,
                              const Eigen::VectorX<Scalar> &reference) {
    const Scalar ref_norm = reference.norm();
    const Scalar diff_norm = (candidate - reference).norm();
    return (ref_norm > Scalar(0)) ? diff_norm / ref_norm : diff_norm;
}

/*!
 * @brief The error floor at the run's fixed rank: the relative Frobenius error
 * of the *best* TT approximation of the dense reference field whose bond ranks
 * do not exceed `rank`.
 *
 * A rank-capped TT-SVD at zero tolerance — the truncation is driven purely by
 * the rank cap, so every bond lands at exactly `rank` (or lower, where the
 * unfolding has no more directions to give). That is the standard
 * quasi-optimal TT approximation — within sqrt(d-1) of the true best rank-r
 * train — and is the fair floor to compare a rank-r cross integrator against.
 *
 * Because the rank is now fixed for the whole run, this depends only on the
 * snapshot, not on which selector produced the state: one floor per snapshot,
 * shared by every curve.
 */
Scalar bestApproximationError(const Eigen::VectorX<Scalar> &reference,
                              const std::vector<Eigen::Index> &sizes,
                              Eigen::Index rank) {
    const TensorTrain<Scalar> best =
        ttFromDenseTensor<Scalar>(reference, sizes, Scalar(0), rank);
    return relativeFrobeniusError(best.toDense(), reference);
}

/*!
 * @brief Writes the bond-`bond` left unfolding of `train` as a plain CSV.
 *
 * This is the matrix a cross step actually hands the selector: rows are the
 * (left-index, mode) pairs, columns the right bond index. Dumping it lets the
 * standard `Tester` pipeline — and therefore the exact two-panel figure of the
 * superconductivity experiment — be pointed at a matrix this PDE produces
 * mid-run, instead of at a static dataset.
 *
 * `MatrixFromFileGenerator` auto-transposes a tall matrix to be wide, which is
 * what a TT unfolding is (many more rows than columns), so it arrives at the
 * selectors in the orientation they require.
 */
void writeUnfoldingCsv(const TensorTrain<Scalar> &train, std::size_t bond,
                       const std::filesystem::path &path) {
    const Eigen::MatrixX<Scalar> &M = train.core(bond).leftUnfolding();
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
 * where the process's main thread gets 8 MB. Eigen's BDCSVD puts sizeable
 * temporaries on the stack, so the TT-SVD of a full field — fine on the main
 * thread — segfaults inside a default-stack worker as soon as the grid grows
 * (n = 64 crashes; n = 32 happens to fit). pthreads is the only portable way
 * to set the stack size, so the pool is built directly on it.
 *
 * `body` must not throw: it runs as a C callback, and an exception escaping it
 * would terminate. Both call sites keep their work inside a try/catch that
 * stores the message instead.
 */
void runOnBigStackThreads(unsigned int count,
                          const std::function<void()> &body) {
    // 64 MB: comfortably above what BDCSVD needs for the unfoldings this
    // experiment produces, and it is address space, not resident memory.
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

    // Fail fast on an unknown algorithm name or a bad parameter: building one
    // selector per spec here turns a typo into an immediate error, before the
    // dense reference trajectory and the floors are computed.
    for (const AlgorithmSpec &spec : opt.algorithms) {
        try {
            factory.create(spec.config);
        } catch (const std::exception &e) {
            std::cerr << "Bad algorithm \"" << spec.display_name
                      << "\": " << e.what() << '\n';
            return 1;
        }
    }

    const Scalar kappa = opt.kappa;

    // Fixed-rank run: both tolerances are zero, so the TT-SVD tail rule never
    // discards anything and `opt.rank` alone decides every bond. Nothing in
    // the run adapts the rank.
    const Scalar atol = Scalar(0);
    const Scalar rtol = Scalar(0);

    // Periodic interior lattice, exactly as the AllenCahn demo: n points
    // spanning [0, 2*pi) with spacing h = 2*pi/n, endpoints excluded so the
    // csc(-x/2) singularities of the initial condition are never sampled.
    const Scalar two_pi = Scalar(2) * Scalar(M_PI);
    const Scalar h = two_pi / static_cast<Scalar>(opt.n);
    const std::vector<Eigen::Index> sizes{opt.n, opt.n, opt.n};
    const std::vector<Scalar> spacings{h, h, h};
    const Grid<Scalar> grid(sizes, {Scalar(0), Scalar(0), Scalar(0)},
                            {two_pi - h, two_pi - h, two_pi - h});

    // Diffusion CFL for an explicit step on the order-`order` periodic
    // Laplacian in 3D, at a conservative 1/4 of the Euler limit. The reaction
    // f - f^3 is non-stiff on the order-1 timescale and does not tighten it.
    const Scalar spectral_factor =
        secondDerivativeSpectralFactor<Scalar>(opt.order);
    const Scalar dt = Scalar(0.25) * Scalar(2) /
                      (kappa * Scalar(sizes.size()) * spectral_factor / (h * h));
    const int n_steps = static_cast<int>(std::ceil(opt.t_end / dt));
    const int save_every = std::max(1, n_steps / std::max(1, opt.snapshots));

    // SSP-RK2 (the midpoint rule): the scheme this experiment is specified on.
    const Scheme<Scalar> scheme =
        Scheme<Scalar>::lowStorageRK({Scalar(0.5), Scalar(1)});

    const auto num_samples = [&opt](Eigen::Index rank, Eigen::Index) {
        return static_cast<Eigen::Index>(
                   std::ceil(opt.width_factor * static_cast<Scalar>(rank))) +
               opt.oversampling;
    };

    const unsigned int n_threads =
        (opt.threads > 0)
            ? static_cast<unsigned int>(opt.threads)
            : std::max(1u, std::thread::hardware_concurrency());

    std::cout << "grid " << opt.n << "^3 (periodic), h = " << h
              << ", dt = " << dt << ", " << n_steps << " steps to t = "
              << opt.t_end << "\nkappa = " << kappa
              << ", fixed rank = " << opt.rank << " (atol = rtol = 0)"
              << ", RK2, stencil order = " << opt.order
              << ", width policy = ceil(" << opt.width_factor << " * r) + "
              << opt.oversampling << ", trials = " << opt.trials
              << ", threads = " << n_threads << '\n';

    const Eigen::VectorX<Scalar> f0_dense = denseFieldFromFunction<Scalar>(
        grid, [](Scalar x1, Scalar x2, Scalar x3) {
            return allenCahnInitial<Scalar>(x1, x2, x3);
        });
    // The initial state enters at the run's fixed rank too, so the integration
    // starts exactly where every later step lives.
    const TensorTrain<Scalar> f0_train =
        ttFromDenseTensor<Scalar>(f0_dense, sizes, rtol, opt.rank);
    std::cout << "initial state: TT ranks " << interiorRanks(f0_train)
              << " (max " << maxRank(f0_train) << ")\n";

    const std::filesystem::path &out_dir = opt.output_path;
    std::filesystem::create_directories(out_dir);

    // ── the dense reference trajectory ────────────────────────────────────────
    // Run once, up front, and keep the field at each snapshot step. Every
    // selector's run is then scored against the same stored trajectory instead
    // of re-integrating the (expensive) full-grid solver nine times over.
    std::cout << "\n==> dense reference trajectory\n";
    std::vector<Eigen::VectorX<Scalar>> reference_fields;
    std::vector<int> reference_steps;
    std::vector<Scalar> reference_times;
    {
        std::vector<Eigen::VectorX<Scalar>> initial_history{f0_dense};
        AllenCahnDenseSolver<Scalar> dense(kappa, sizes, spacings, scheme, dt,
                                           std::move(initial_history),
                                           opt.order);
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

    // The mid-run unfolding dump: TT-SVD the reference field at the middle
    // snapshot at the run's fixed rank, then write one interior bond's left
    // unfolding. This is a matrix the cross solver genuinely selects columns
    // from, at a moment when the dynamics are fully developed.
    {
        const std::size_t mid = reference_fields.size() / 2;
        const TensorTrain<Scalar> mid_train = ttFromDenseTensor<Scalar>(
            reference_fields[mid], sizes, rtol, opt.rank);
        std::cout << "\nmid-run state (t = " << reference_times[mid]
                  << "): TT ranks " << interiorRanks(mid_train) << '\n';
        writeUnfoldingCsv(mid_train, /*bond=*/1,
                          out_dir / "allen_cahn_unfolding.csv");
    }

    // ── the error floor, once per snapshot ────────────────────────────────────
    // The rank is fixed for the whole run, so the best rank-r approximation of
    // a snapshot no longer depends on which selector produced the state: one
    // floor per snapshot serves every curve. (With adaptive ranks this had to
    // be memoized per rank profile.) Each is an independent TT-SVD of a full
    // field, so they are computed in parallel.
    std::cout << "\n==> best rank-" << opt.rank << " floors\n";
    std::vector<Scalar> best_errors(reference_fields.size());
    {
        std::atomic<std::size_t> next{0};
        std::mutex error_mutex;
        std::string failure;
        const unsigned int workers = std::min<unsigned int>(
            n_threads, static_cast<unsigned int>(best_errors.size()));
        runOnBigStackThreads(workers, [&] {
            try {
                for (std::size_t i = next++; i < best_errors.size();
                     i = next++) {
                    best_errors[i] = bestApproximationError(
                        reference_fields[i], sizes, opt.rank);
                }
            } catch (const std::exception &e) {
                const std::lock_guard<std::mutex> lock(error_mutex);
                if (failure.empty()) {
                    failure = e.what();
                }
            }
        });
        if (!failure.empty()) {
            std::cerr << "Failed to compute the floors: " << failure << '\n';
            return 1;
        }
        std::cout << "   " << best_errors.size() << " floors computed\n";
    }

    // ── per-selector runs, in parallel ────────────────────────────────────────
    // Each (selector, seed) pair is a completely independent integration —
    // nothing is shared but the read-only reference trajectory and the
    // precomputed floors — so the pairs are flattened into one job list and
    // handed to a thread pool. Each job writes its rows into its own slot of
    // `rows`, which is then emitted in job order: the CSV is byte-identical
    // whatever order the threads happen to finish in.
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
        const unsigned int workers =
            std::min<unsigned int>(n_threads, static_cast<unsigned int>(
                                                  std::max<std::size_t>(
                                                      jobs.size(), 1)));
        runOnBigStackThreads(workers, [&] {
          try {
                for (std::size_t j = next++; j < jobs.size(); j = next++) {
                    const Job &job = jobs[j];

                    std::vector<TensorTrain<Scalar>> initial_history{f0_train};
                    Solver<Scalar> solver(
                        std::move(initial_history),
                        std::make_unique<AllenCahnRhs<Scalar>>(
                            kappa, sizes, spacings, opt.order),
                        scheme, dt, factory.create(job.config),
                        atol, rtol, num_samples, /*boundary_condition=*/nullptr,
                        /*warmup_steps=*/0, opt.rank);

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
                        const Scalar error = relativeFrobeniusError(
                            state.toDense(), reference_fields[snapshot]);

                        // The rank profile is 'x'-separated, so it is quoted:
                        // it is the last column today, but a quoted field
                        // survives a reordering.
                        out << job.display_name << ',' << job.sample << ','
                            << step << ',' << reference_times[snapshot] << ','
                            << error << ',' << best_errors[snapshot] << ','
                            << maxRank(state) << ",\"" << interiorRanks(state)
                            << "\"\n";
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

    std::ofstream csv(out_dir / "allen_cahn_errors.csv");
    if (!csv) {
        throw std::runtime_error("cannot write the errors CSV");
    }
    csv << "algorithm,sample,step,t,error,best_error,max_rank,ranks\n";
    for (const std::string &row : rows) {
        csv << row;
    }

    std::cout << "\nwrote " << (out_dir / "allen_cahn_errors.csv") << '\n';
    return 0;
}
