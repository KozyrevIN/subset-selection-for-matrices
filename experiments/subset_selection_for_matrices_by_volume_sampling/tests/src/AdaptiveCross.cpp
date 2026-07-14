#include <doctest/doctest.h>

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include <MatSubset/DominantSelector.h>
#include <MatSubset/SelectorBase.h>

#include <AcousticEquation/AcousticRhs.h>
#include <TTCrossSolver/AdaptiveSolver.h>
#include <TTCrossSolver/FiberEvaluator.h>
#include <TTCrossSolver/Solver.h>
#include <TTCrossSolver/TensorFibers.h>
#include <TTCrossSolver/TensorTrain.h>
#include <TTCrossSolver/TensorTrainCore.h>

using MatSubset::Experiments::AdaptiveSolver;
using MatSubset::Experiments::FiberEvaluatorBase;
using MatSubset::Experiments::FiberIndices;
using MatSubset::Experiments::MaskBoundaryCondition;
using MatSubset::Experiments::RhsBase;
using MatSubset::Experiments::Scheme;
using MatSubset::Experiments::TensorFibers;
using MatSubset::Experiments::TensorTrain;
using MatSubset::Experiments::TensorTrainCore;
using MatSubset::Experiments::TrainFiberEvaluator;

namespace {

// Deterministic, well-conditioned left unfolding (r0 * n) x r1.
template <typename Scalar>
Eigen::MatrixX<Scalar> makeUnfolding(Eigen::Index r0, Eigen::Index n,
                                     Eigen::Index r1, Eigen::Index salt) {
    Eigen::MatrixX<Scalar> A(r0 * n, r1);
    for (Eigen::Index i = 0; i < A.rows(); ++i) {
        for (Eigen::Index j = 0; j < A.cols(); ++j) {
            A(i, j) = static_cast<Scalar>(1 + (i * 7 + j * 3 + salt * 5) % 11);
        }
    }
    return A;
}

// Builds a 3-core train with modes (3, 4, 3), ranks (2, 2) and the given salt.
template <typename Scalar> TensorTrain<Scalar> makeTrain(Eigen::Index salt) {
    std::vector<TensorTrainCore<Scalar>> cores;
    cores.emplace_back(makeUnfolding<Scalar>(1, 3, 2, salt), 3);
    cores.emplace_back(makeUnfolding<Scalar>(2, 4, 2, salt + 1), 4);
    cores.emplace_back(makeUnfolding<Scalar>(2, 3, 1, salt + 2), 3);
    return TensorTrain<Scalar>(std::move(cores));
}

template <typename Scalar> Scalar checkTol() {
    return std::is_same_v<Scalar, float> ? Scalar(1e-3) : Scalar(1e-9);
}

template <typename Scalar>
std::unique_ptr<MatSubset::SelectorBase<Scalar>> makeSelector() {
    return std::make_unique<MatSubset::DominantSelector<Scalar>>(Scalar(1));
}

// A nested 3-core skeleton of width 2 at every bond, with sequential
// (non-adapted) indices: a valid but deliberately uninformed warm start.
FiberIndices makeSequentialSkeleton() {
    using Level = FiberIndices::Level;
    std::vector<Level> left(3), right(3);

    left[0] = Level({0, 1}, {-1, -1});
    left[1] = Level({0, 1}, {0, 1});
    // left[2] stays empty (no left set at the last bond).

    right[2] = Level({0}, {-1}); // right root
    right[1] = Level({0, 1}, {0, 0});
    right[0] = Level({0, 1}, {0, 1});

    return FiberIndices(std::move(left), std::move(right));
}

// F(y, t) = lambda * y with the slab-wise evaluator implemented, for the
// adaptive solver's exact scalar-recurrence checks.
template <typename Scalar> class ScaleRhs : public RhsBase<Scalar> {
  public:
    explicit ScaleRhs(Scalar lambda) : lambda(lambda) {}

    [[nodiscard]] std::unique_ptr<FiberEvaluatorBase<Scalar>>
    makeEvaluator(const TensorTrain<Scalar> &state, Scalar) const override {
        return std::make_unique<TrainFiberEvaluator<Scalar>>(state, lambda);
    }

    [[nodiscard]] TensorTrain<Scalar>
    evaluateTrain(const TensorTrain<Scalar> &state, Scalar) const override {
        return lambda * state;
    }

  private:
    Scalar lambda;
};

// A rank-1 all-constant mask train (factor in the first core).
template <typename Scalar>
TensorTrain<Scalar> makeConstantMask(const std::vector<Eigen::Index> &sizes,
                                     Scalar factor) {
    std::vector<TensorTrainCore<Scalar>> cores;
    for (std::size_t k = 0; k < sizes.size(); ++k) {
        Eigen::MatrixX<Scalar> ones =
            Eigen::MatrixX<Scalar>::Ones(sizes[k], 1);
        cores.emplace_back((k == 0) ? Eigen::MatrixX<Scalar>(factor * ones)
                                    : ones,
                           sizes[k]);
    }
    return TensorTrain<Scalar>(std::move(cores));
}

} // namespace

TEST_CASE_TEMPLATE("TensorTrain::atFiber matches atFibers slab by slab",
                   Scalar, float, double) {
    TensorTrain<Scalar> train = makeTrain<Scalar>(0);
    auto skeleton =
        std::make_shared<const FiberIndices>(makeSequentialSkeleton());

    TensorFibers<Scalar> fibers = train.atFibers(skeleton);
    for (std::size_t k = 0; k < train.order(); ++k) {
        const Eigen::MatrixX<Scalar> slab =
            train.atFiber(k, *skeleton).leftUnfolding();
        CHECK(slab.rows() == fibers.core(k).leftUnfolding().rows());
        CHECK(slab.cols() == fibers.core(k).leftUnfolding().cols());
        CHECK((slab - fibers.core(k).leftUnfolding()).norm() <=
              checkTol<Scalar>() * fibers.core(k).leftUnfolding().norm());
    }
}

TEST_CASE_TEMPLATE(
    "crossInterpolate recovers a train from a sequential warm start", Scalar,
    float, double) {
    TensorTrain<Scalar> train = makeTrain<Scalar>(3);
    Eigen::MatrixX<Scalar> dense = train.toDense();

    auto selector = makeSelector<Scalar>();
    TrainFiberEvaluator<Scalar> evaluator(train);
    auto [rebuilt, skeleton] = TensorTrain<Scalar>::crossInterpolate(
        evaluator, makeSequentialSkeleton(), selector, Scalar(0),
        checkTol<Scalar>(), /*oversample=*/1, /*rounds=*/2);

    REQUIRE(rebuilt.order() == train.order());
    CHECK(rebuilt.modeSizes() == train.modeSizes());
    CHECK((rebuilt.toDense() - dense).norm() <
          Scalar(100) * checkTol<Scalar>() * dense.norm());

    // The forward sweep selects each left level after the truncation that
    // fixes the core, so left widths are rank + oversample, clamped to the
    // candidate count.
    const std::vector<Eigen::Index> ranks = rebuilt.ranks();
    for (std::size_t k = 0; k + 1 < rebuilt.order(); ++k) {
        const Eigen::Index candidates =
            static_cast<Eigen::Index>(skeleton->leftFiberCount(k)) *
            rebuilt.core(k).modeSize();
        CHECK(static_cast<Eigen::Index>(skeleton->leftLevel(k).size()) ==
              std::min(ranks[k + 1] + 1, candidates));
    }
}

TEST_CASE_TEMPLATE("crossInterpolate round-trips through atFibers", Scalar,
                   float, double) {
    // The returned skeleton must evaluate the returned train exactly (the
    // left levels are consistent with the final cores by construction).
    TensorTrain<Scalar> train = makeTrain<Scalar>(7);

    auto selector = makeSelector<Scalar>();
    TrainFiberEvaluator<Scalar> evaluator(train);
    auto [rebuilt, skeleton] = TensorTrain<Scalar>::crossInterpolate(
        evaluator, makeSequentialSkeleton(), selector, Scalar(0),
        checkTol<Scalar>(), /*oversample=*/1);

    TensorFibers<Scalar> fibers = rebuilt.atFibers(skeleton);
    TensorTrain<Scalar> again(fibers, Scalar(0), checkTol<Scalar>());
    CHECK((again.toDense() - rebuilt.toDense()).norm() <
          Scalar(100) * checkTol<Scalar>() * rebuilt.toDense().norm());
}

TEST_CASE_TEMPLATE("AdaptiveSolver forward Euler matches the discrete "
                   "solution",
                   Scalar, float, double) {
    const Scalar lambda = Scalar(-0.5);
    const Scalar dt = Scalar(0.1);
    const int n_steps = 10;

    auto y0 = makeTrain<Scalar>(0);
    Eigen::MatrixX<Scalar> dense0 = y0.toDense();

    std::vector<TensorTrain<Scalar>> init;
    init.push_back(std::move(y0));
    AdaptiveSolver<Scalar> solver(
        std::move(init), std::make_unique<ScaleRhs<Scalar>>(lambda),
        Scheme<Scalar>::forwardEuler(), dt, makeSelector<Scalar>(), Scalar(0),
        checkTol<Scalar>(), /*oversample=*/1);

    // Euler on y' = lambda*y multiplies by (1 + lambda*dt) each step, exactly.
    Scalar factor = Scalar(1);
    for (int n = 0; n < n_steps; ++n) {
        solver.step();
        factor *= Scalar(1) + lambda * dt;
    }

    CHECK(static_cast<double>(solver.time()) ==
          doctest::Approx(n_steps * static_cast<double>(dt)));
    CHECK((solver.getState().toDense() - factor * dense0).norm() <
          Scalar(100) * checkTol<Scalar>() * dense0.norm());
}

TEST_CASE_TEMPLATE(
    "AdaptiveSolver second-order leapfrog matches the discrete recurrence",
    Scalar, float, double) {
    // y'' = -omega^2 * y, the wave-equation prototype; exercises history
    // terms sampled through the combo evaluator.
    const Scalar omega = Scalar(1);
    const Scalar dt = Scalar(0.1);
    const int n_steps = 20;

    auto y0 = makeTrain<Scalar>(0);
    Eigen::MatrixX<Scalar> dense0 = y0.toDense();

    std::vector<TensorTrain<Scalar>> init;
    init.push_back(makeTrain<Scalar>(0)); // y_{-1} = y_0
    init.push_back(std::move(y0));
    AdaptiveSolver<Scalar> solver(
        std::move(init), std::make_unique<ScaleRhs<Scalar>>(-omega * omega),
        Scheme<Scalar>::leapfrogSecondOrder(), dt, makeSelector<Scalar>(),
        Scalar(0), checkTol<Scalar>(), /*oversample=*/1);

    // y_n = a_n * y_0 with a_{n+1} = (2 - omega^2*dt^2)*a_n - a_{n-1}.
    Scalar a_prev = Scalar(1);
    Scalar a = Scalar(1);
    for (int n = 0; n < n_steps; ++n) {
        solver.step();
        const Scalar a_next =
            (Scalar(2) - omega * omega * dt * dt) * a - a_prev;
        a_prev = a;
        a = a_next;
    }

    CHECK((solver.getState().toDense() - a * dense0).norm() <
          Scalar(100) * checkTol<Scalar>() * dense0.norm());
}

TEST_CASE_TEMPLATE("AdaptiveSolver applies the boundary condition once per "
                   "step through the wrapped evaluator",
                   Scalar, float, double) {
    const Scalar dt = Scalar(0.25);
    const int n_steps = 3;

    auto y0 = makeTrain<Scalar>(0);
    Eigen::MatrixX<Scalar> dense0 = y0.toDense();

    std::vector<TensorTrain<Scalar>> init;
    init.push_back(std::move(y0));
    // Zero rhs + halving mask: y_n = y_0 / 2^n exactly. The two-stage scheme
    // checks the mask wraps the final stage only.
    AdaptiveSolver<Scalar> solver(
        std::move(init), std::make_unique<ScaleRhs<Scalar>>(Scalar(0)),
        Scheme<Scalar>::lowStorageRK({Scalar(0.5), Scalar(1)}), dt,
        makeSelector<Scalar>(), Scalar(0), checkTol<Scalar>(),
        /*oversample=*/1,
        std::make_unique<MaskBoundaryCondition<Scalar>>(
            makeConstantMask<Scalar>({3, 4, 3}, Scalar(0.5))));

    Scalar factor = Scalar(1);
    for (int n = 0; n < n_steps; ++n) {
        solver.step();
        factor *= Scalar(0.5);
    }

    CHECK((solver.getState().toDense() - factor * dense0).norm() <
          Scalar(100) * checkTol<Scalar>() * dense0.norm());
}

TEST_CASE_TEMPLATE("AdaptiveSolver warm-up hands off to the adaptive path",
                   Scalar, float, double) {
    const Scalar lambda = Scalar(-0.3);
    const Scalar dt = Scalar(0.05);
    const int n_steps = 10;

    auto y0 = makeTrain<Scalar>(0);
    Eigen::MatrixX<Scalar> dense0 = y0.toDense();

    std::vector<TensorTrain<Scalar>> init;
    init.push_back(makeTrain<Scalar>(0));
    init.push_back(std::move(y0));
    AdaptiveSolver<Scalar> solver(
        std::move(init), std::make_unique<ScaleRhs<Scalar>>(lambda),
        Scheme<Scalar>::leapfrog(), dt, makeSelector<Scalar>(), Scalar(0),
        checkTol<Scalar>(), /*oversample=*/1, /*boundary=*/nullptr,
        /*warmup_steps=*/4);

    Scalar a_prev = Scalar(1);
    Scalar a = Scalar(1);
    for (int n = 0; n < n_steps; ++n) {
        solver.step();
        const Scalar a_next = a_prev + Scalar(2) * dt * lambda * a;
        a_prev = a;
        a = a_next;
    }

    CHECK((solver.getState().toDense() - a * dense0).norm() <
          Scalar(100) * checkTol<Scalar>() * dense0.norm());
}

TEST_CASE_TEMPLATE("AcousticRhs::makeEvaluator matches evaluate slab-wise",
                   Scalar, float, double) {
    using MatSubset::Experiments::AcousticRhs;

    const std::vector<Eigen::Index> sizes{3, 4, 3};
    const std::vector<Scalar> spacings{Scalar(0.5), Scalar(1), Scalar(0.5)};

    // A layered (rank-1) medium and a localized separable source.
    auto speed = makeConstantMask<Scalar>(sizes, Scalar(2));
    auto source = makeTrain<Scalar>(5);
    const auto envelope = [](Scalar t) { return Scalar(1) + t; };

    AcousticRhs<Scalar> rhs(speed, source, envelope, sizes, spacings);

    // A state with structure, and its skeleton via the fixed-skeleton path.
    TensorTrain<Scalar> state = makeTrain<Scalar>(1);
    state.leftOrthogonalize();
    auto selector = makeSelector<Scalar>();
    TensorFibers<Scalar> state_fibers =
        state.selectIndices(selector, Scalar(0), checkTol<Scalar>());

    const Scalar t = Scalar(0.3);
    TensorFibers<Scalar> reference = rhs.evaluate(state, state_fibers, t);
    std::unique_ptr<FiberEvaluatorBase<Scalar>> evaluator =
        rhs.makeEvaluator(state, t);
    REQUIRE(evaluator);
    CHECK(evaluator->modeSizes() == sizes);

    for (std::size_t k = 0; k < state.order(); ++k) {
        const Eigen::MatrixX<Scalar> slab =
            evaluator->atFiber(k, *state_fibers.skeleton()).leftUnfolding();
        CHECK((slab - reference.core(k).leftUnfolding()).norm() <=
              Scalar(10) * checkTol<Scalar>() *
                  (reference.core(k).leftUnfolding().norm() + Scalar(1)));
    }
}
