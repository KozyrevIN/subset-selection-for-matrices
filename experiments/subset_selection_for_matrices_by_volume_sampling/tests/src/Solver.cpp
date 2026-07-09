#include <doctest/doctest.h>

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include <MatSubset/DominantSelector.h>
#include <MatSubset/SelectorBase.h>

#include <TTCrossSolver/Solver.h>
#include <TTCrossSolver/TensorFibers.h>
#include <TTCrossSolver/TensorTrain.h>
#include <TTCrossSolver/TensorTrainCore.h>

using MatSubset::Experiments::RhsBase;
using MatSubset::Experiments::Scheme;
using MatSubset::Experiments::Solver;
using MatSubset::Experiments::TensorFibers;
using MatSubset::Experiments::TensorTrain;
using MatSubset::Experiments::TensorTrainCore;

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

// Builds a 3-core train with the given bond ranks and mode sizes.
template <typename Scalar>
TensorTrain<Scalar> makeTrain(Eigen::Index n0, Eigen::Index n1, Eigen::Index n2,
                              Eigen::Index r1, Eigen::Index r2) {
    std::vector<TensorTrainCore<Scalar>> cores;
    cores.emplace_back(makeUnfolding<Scalar>(1, n0, r1, 0), n0);
    cores.emplace_back(makeUnfolding<Scalar>(r1, n1, r2, 1), n1);
    cores.emplace_back(makeUnfolding<Scalar>(r2, n2, 1, 2), n2);
    return TensorTrain<Scalar>(std::move(cores));
}

template <typename Scalar> Scalar checkTol() {
    return std::is_same_v<Scalar, float> ? Scalar(1e-3) : Scalar(1e-9);
}

template <typename Scalar>
std::unique_ptr<MatSubset::SelectorBase<Scalar>> makeSelector() {
    return std::make_unique<MatSubset::DominantSelector<Scalar>>(Scalar(1));
}

// F(y, t) = lambda * y: the discrete solutions are exact scalar recurrences,
// so every scheme can be checked against them to roundoff/truncation error.
template <typename Scalar> class ScaleRhs : public RhsBase<Scalar> {
  public:
    explicit ScaleRhs(Scalar lambda) : lambda(lambda) {}

    [[nodiscard]] TensorFibers<Scalar>
    evaluate(const TensorTrain<Scalar> &,
             const TensorFibers<Scalar> &state_fibers, Scalar) const override {
        return lambda * state_fibers;
    }

  private:
    Scalar lambda;
};

// F = 0, but records every stage time it is called with.
template <typename Scalar> class TimeLogRhs : public RhsBase<Scalar> {
  public:
    explicit TimeLogRhs(std::vector<Scalar> &log) : log(&log) {}

    [[nodiscard]] TensorFibers<Scalar>
    evaluate(const TensorTrain<Scalar> &,
             const TensorFibers<Scalar> &state_fibers,
             Scalar t) const override {
        log->push_back(t);
        return Scalar(0) * state_fibers;
    }

  private:
    std::vector<Scalar> *log;
};

} // namespace

TEST_CASE("Scheme named constructors") {
    auto euler = Scheme<double>::forwardEuler();
    CHECK(euler.stages.size() == 1);
    CHECK(euler.time_order == 1);
    CHECK(euler.history == 1);
    CHECK(euler.stages[0].history_weights == std::vector<double>{1.0});
    CHECK(euler.stages[0].rhs_weight == 1.0);
    CHECK(euler.stages[0].rhs_time_offset == 0.0);

    auto rk = Scheme<double>::lowStorageRK({0.5, 1.0});
    CHECK(rk.stages.size() == 2);
    CHECK(rk.history == 1);
    CHECK(rk.stages[0].rhs_weight == 0.5);
    CHECK(rk.stages[0].rhs_time_offset == 0.0);
    CHECK(rk.stages[1].rhs_weight == 1.0);
    CHECK(rk.stages[1].rhs_time_offset == 0.5);

    auto lf = Scheme<double>::leapfrog();
    CHECK(lf.stages.size() == 1);
    CHECK(lf.time_order == 1);
    CHECK(lf.history == 2);
    CHECK(lf.stages[0].history_weights == std::vector<double>{0.0, 1.0});
    CHECK(lf.stages[0].rhs_weight == 2.0);

    auto lf2 = Scheme<double>::leapfrogSecondOrder();
    CHECK(lf2.stages.size() == 1);
    CHECK(lf2.time_order == 2);
    CHECK(lf2.history == 2);
    CHECK(lf2.stages[0].history_weights == std::vector<double>{2.0, -1.0});
    CHECK(lf2.stages[0].rhs_weight == 1.0);
}

TEST_CASE_TEMPLATE("Solver exposes the initial state and time", Scalar, float,
                   double) {
    auto y0 = makeTrain<Scalar>(3, 4, 2, 2, 2);
    Eigen::MatrixX<Scalar> dense0 = y0.toDense();

    std::vector<TensorTrain<Scalar>> init;
    init.push_back(std::move(y0));
    Solver<Scalar> solver(
        std::move(init), std::make_unique<ScaleRhs<Scalar>>(Scalar(1)),
        Scheme<Scalar>::forwardEuler(), Scalar(0.1), makeSelector<Scalar>(),
        Scalar(0), checkTol<Scalar>());

    CHECK(solver.time() == Scalar(0));
    // The constructor orthogonalizes but does not change the tensor.
    CHECK((solver.getState().toDense() - dense0).norm() <
          checkTol<Scalar>() * dense0.norm());
}

TEST_CASE_TEMPLATE("Solver forward Euler matches the discrete solution", Scalar,
                   float, double) {
    const Scalar lambda = Scalar(-0.5);
    const Scalar dt = Scalar(0.1);
    const int n_steps = 10;

    auto y0 = makeTrain<Scalar>(3, 4, 2, 2, 2);
    Eigen::MatrixX<Scalar> dense0 = y0.toDense();

    std::vector<TensorTrain<Scalar>> init;
    init.push_back(std::move(y0));
    Solver<Scalar> solver(
        std::move(init), std::make_unique<ScaleRhs<Scalar>>(lambda),
        Scheme<Scalar>::forwardEuler(), dt, makeSelector<Scalar>(), Scalar(0),
        checkTol<Scalar>());

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

TEST_CASE_TEMPLATE("Solver midpoint RK matches the discrete solution", Scalar,
                   float, double) {
    const Scalar lambda = Scalar(-0.4);
    const Scalar dt = Scalar(0.1);
    const int n_steps = 10;

    auto y0 = makeTrain<Scalar>(3, 4, 2, 2, 2);
    Eigen::MatrixX<Scalar> dense0 = y0.toDense();

    std::vector<TensorTrain<Scalar>> init;
    init.push_back(std::move(y0));
    Solver<Scalar> solver(
        std::move(init), std::make_unique<ScaleRhs<Scalar>>(lambda),
        Scheme<Scalar>::lowStorageRK({Scalar(0.5), Scalar(1)}), dt,
        makeSelector<Scalar>(), Scalar(0), checkTol<Scalar>());

    // Midpoint on y' = lambda*y multiplies by 1 + z + z^2/2 (z = lambda*dt)
    // each step, exactly.
    const Scalar z = lambda * dt;
    Scalar factor = Scalar(1);
    for (int n = 0; n < n_steps; ++n) {
        solver.step();
        factor *= Scalar(1) + z + z * z / Scalar(2);
    }

    CHECK((solver.getState().toDense() - factor * dense0).norm() <
          Scalar(100) * checkTol<Scalar>() * dense0.norm());
}

TEST_CASE_TEMPLATE("Solver leapfrog matches the discrete recurrence", Scalar,
                   float, double) {
    const Scalar lambda = Scalar(-0.3);
    const Scalar dt = Scalar(0.05);
    const int n_steps = 10;

    auto y0 = makeTrain<Scalar>(3, 4, 2, 2, 2);
    Eigen::MatrixX<Scalar> dense0 = y0.toDense();

    // Initial history {y_{-1}, y_0} with y_{-1} = y_0 (a_{-1} = a_0 = 1 in
    // the scalar recurrence below).
    std::vector<TensorTrain<Scalar>> init;
    init.push_back(makeTrain<Scalar>(3, 4, 2, 2, 2));
    init.push_back(std::move(y0));
    Solver<Scalar> solver(
        std::move(init), std::make_unique<ScaleRhs<Scalar>>(lambda),
        Scheme<Scalar>::leapfrog(), dt, makeSelector<Scalar>(), Scalar(0),
        checkTol<Scalar>());

    // Every state stays proportional to y_0: y_n = a_n * y_0 with
    // a_{n+1} = a_{n-1} + 2*dt*lambda*a_n.
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

TEST_CASE_TEMPLATE(
    "Solver second-order leapfrog matches the discrete recurrence", Scalar,
    float, double) {
    // y'' = -omega^2 * y, the wave-equation prototype.
    const Scalar omega = Scalar(1);
    const Scalar dt = Scalar(0.1);
    const int n_steps = 20;

    auto y0 = makeTrain<Scalar>(3, 4, 2, 2, 2);
    Eigen::MatrixX<Scalar> dense0 = y0.toDense();

    std::vector<TensorTrain<Scalar>> init;
    init.push_back(makeTrain<Scalar>(3, 4, 2, 2, 2)); // y_{-1} = y_0
    init.push_back(std::move(y0));
    Solver<Scalar> solver(
        std::move(init), std::make_unique<ScaleRhs<Scalar>>(-omega * omega),
        Scheme<Scalar>::leapfrogSecondOrder(), dt, makeSelector<Scalar>(),
        Scalar(0), checkTol<Scalar>());

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

TEST_CASE_TEMPLATE("Solver passes the stage times to the rhs", Scalar, float,
                   double) {
    const Scalar dt = Scalar(0.25);

    std::vector<Scalar> log;
    auto y0 = makeTrain<Scalar>(3, 4, 2, 2, 2);
    Eigen::MatrixX<Scalar> dense0 = y0.toDense();

    std::vector<TensorTrain<Scalar>> init;
    init.push_back(std::move(y0));
    Solver<Scalar> solver(
        std::move(init), std::make_unique<TimeLogRhs<Scalar>>(log),
        Scheme<Scalar>::lowStorageRK({Scalar(0.5), Scalar(1)}), dt,
        makeSelector<Scalar>(), Scalar(0), checkTol<Scalar>());

    solver.step();
    // Stage 1 at t_0, stage 2 at t_0 + dt/2 (theta_2 = c_1 = 1/2).
    REQUIRE(log.size() == 2);
    CHECK(static_cast<double>(log[0]) == doctest::Approx(0.0));
    CHECK(static_cast<double>(log[1]) ==
          doctest::Approx(static_cast<double>(dt) / 2));

    solver.step();
    REQUIRE(log.size() == 4);
    CHECK(static_cast<double>(log[2]) ==
          doctest::Approx(static_cast<double>(dt)));
    CHECK(static_cast<double>(log[3]) ==
          doctest::Approx(1.5 * static_cast<double>(dt)));

    // A zero rhs leaves the state unchanged.
    CHECK((solver.getState().toDense() - dense0).norm() <
          Scalar(100) * checkTol<Scalar>() * dense0.norm());
}
