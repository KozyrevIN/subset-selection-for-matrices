#include <doctest/doctest.h>

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include <MatSubset/DominantSelector.h>
#include <MatSubset/SelectorBase.h>

#include <AcousticEquation/AcousticRhs.h>
#include <TTCrossSolver/Solver.h>
#include <TTCrossSolver/TensorFibers.h>
#include <TTCrossSolver/TensorTrain.h>
#include <TTCrossSolver/TensorTrainCore.h>

using MatSubset::Experiments::AcousticRhs;
using MatSubset::Experiments::FiberIndices;
using MatSubset::Experiments::makeCerjanMask;
using MatSubset::Experiments::makeLaplacianOperator;
using MatSubset::Experiments::MaskBoundaryCondition;
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

// Builds a 3-core train with modes (3, 4, 2), ranks (2, 2) and the given salt.
template <typename Scalar> TensorTrain<Scalar> makeTrain(Eigen::Index salt) {
    std::vector<TensorTrainCore<Scalar>> cores;
    cores.emplace_back(makeUnfolding<Scalar>(1, 3, 2, salt), 3);
    cores.emplace_back(makeUnfolding<Scalar>(2, 4, 2, salt + 1), 4);
    cores.emplace_back(makeUnfolding<Scalar>(2, 2, 1, salt + 2), 2);
    return TensorTrain<Scalar>(std::move(cores));
}

// Builds a rank-1 2-core train from two mode vectors.
template <typename Scalar>
TensorTrain<Scalar> makeRank1(const Eigen::VectorX<Scalar> &v0,
                              const Eigen::VectorX<Scalar> &v1) {
    std::vector<TensorTrainCore<Scalar>> cores;
    cores.emplace_back(Eigen::MatrixX<Scalar>(v0), v0.size());
    cores.emplace_back(Eigen::MatrixX<Scalar>(v1), v1.size());
    return TensorTrain<Scalar>(std::move(cores));
}

template <typename Scalar> Scalar checkTol() {
    return std::is_same_v<Scalar, float> ? Scalar(1e-3) : Scalar(1e-9);
}

template <typename Scalar>
std::unique_ptr<MatSubset::SelectorBase<Scalar>> makeSelector() {
    return std::make_unique<MatSubset::DominantSelector<Scalar>>(Scalar(1));
}

// The 1D Dirichlet stencil tridiag(1, -2, 1) / h^2, mirroring the operator.
template <typename Scalar>
Eigen::MatrixX<Scalar> denseStencil(Eigen::Index n, Scalar h) {
    Eigen::MatrixX<Scalar> L = Eigen::MatrixX<Scalar>::Zero(n, n);
    const Scalar w = Scalar(1) / (h * h);
    for (Eigen::Index i = 0; i < n; ++i) {
        L(i, i) = Scalar(-2) * w;
        if (i > 0) {
            L(i, i - 1) = w;
        }
        if (i + 1 < n) {
            L(i, i + 1) = w;
        }
    }
    return L;
}

} // namespace

TEST_CASE_TEMPLATE("makeLaplacianOperator matches the dense Kronecker sum",
                   Scalar, float, double) {
    const std::vector<Eigen::Index> sizes{3, 4, 2};
    const std::vector<Scalar> spacings{Scalar(0.5), Scalar(1), Scalar(2)};

    TensorTrain<Scalar> lap = makeLaplacianOperator<Scalar>(sizes, spacings);
    CHECK(lap.ranks() == std::vector<Eigen::Index>{1, 2, 2, 1});

    // Dense Laplacian, first-mode-fastest flattening i0 + n0*i1 + n0*n1*i2:
    // D[(i), (j)] = sum_k L_k(i_k, j_k) * prod_{m != k} delta(i_m, j_m).
    const Eigen::Index N = 3 * 4 * 2;
    std::vector<Eigen::MatrixX<Scalar>> L1d;
    for (std::size_t k = 0; k < sizes.size(); ++k) {
        L1d.push_back(denseStencil<Scalar>(sizes[k], spacings[k]));
    }
    Eigen::MatrixX<Scalar> D = Eigen::MatrixX<Scalar>::Zero(N, N);
    for (Eigen::Index i0 = 0; i0 < 3; ++i0) {
        for (Eigen::Index i1 = 0; i1 < 4; ++i1) {
            for (Eigen::Index i2 = 0; i2 < 2; ++i2) {
                for (Eigen::Index j0 = 0; j0 < 3; ++j0) {
                    for (Eigen::Index j1 = 0; j1 < 4; ++j1) {
                        for (Eigen::Index j2 = 0; j2 < 2; ++j2) {
                            const Eigen::Index row = i0 + 3 * i1 + 12 * i2;
                            const Eigen::Index col = j0 + 3 * j1 + 12 * j2;
                            Scalar v = Scalar(0);
                            if (i1 == j1 && i2 == j2) {
                                v += L1d[0](i0, j0);
                            }
                            if (i0 == j0 && i2 == j2) {
                                v += L1d[1](i1, j1);
                            }
                            if (i0 == j0 && i1 == j1) {
                                v += L1d[2](i2, j2);
                            }
                            D(row, col) = v;
                        }
                    }
                }
            }
        }
    }

    // Apply both to the same TT vector and compare densely.
    auto y = makeTrain<Scalar>(0);
    Eigen::MatrixX<Scalar> expected = D * y.toDense();
    Eigen::MatrixX<Scalar> got = lap.zip(y, sizes, sizes).toDense();
    REQUIRE(got.rows() == expected.rows());
    CHECK((got - expected).norm() <
          Scalar(100) * checkTol<Scalar>() * expected.norm());
}

TEST_CASE_TEMPLATE("makeLaplacianOperator handles a single dimension", Scalar,
                   float, double) {
    const std::vector<Eigen::Index> sizes{5};
    const std::vector<Scalar> spacings{Scalar(0.25)};

    TensorTrain<Scalar> lap = makeLaplacianOperator<Scalar>(sizes, spacings);
    CHECK(lap.ranks() == std::vector<Eigen::Index>{1, 1});

    std::vector<TensorTrainCore<Scalar>> cores;
    cores.emplace_back(makeUnfolding<Scalar>(1, 5, 1, 3), 5);
    TensorTrain<Scalar> y(std::move(cores));

    Eigen::MatrixX<Scalar> expected =
        denseStencil<Scalar>(5, spacings[0]) * y.toDense();
    Eigen::MatrixX<Scalar> got = lap.zip(y, sizes, sizes).toDense();
    CHECK((got - expected).norm() <
          Scalar(100) * checkTol<Scalar>() * expected.norm());
}

TEST_CASE_TEMPLATE("AcousticRhs evaluates c^2 (lap + source) on the skeleton",
                   Scalar, float, double) {
    using Level = FiberIndices::Level;

    const std::vector<Eigen::Index> sizes{3, 4, 2};
    const std::vector<Scalar> spacings{Scalar(0.5), Scalar(1), Scalar(2)};

    auto y = makeTrain<Scalar>(0);
    auto c = makeTrain<Scalar>(10);
    auto s = makeTrain<Scalar>(20);
    const auto f = [](Scalar t) { return Scalar(1) + Scalar(2) * t; };
    const Scalar t = Scalar(0.3);

    AcousticRhs<Scalar> rhs(c, s, f, sizes, spacings);

    // A nested skeleton with known multi-index tuples (the layout used by the
    // TensorTrain::atFibers tests).
    std::vector<Eigen::Index> L0 = {0, 2};
    std::vector<std::pair<Eigen::Index, Eigen::Index>> L1 = {{0, 1}, {2, 3}};
    std::vector<std::pair<Eigen::Index, Eigen::Index>> R0 = {{0, 0}, {2, 1}};
    std::vector<Eigen::Index> R1 = {0, 1};

    std::vector<Level> left(3), right(3);
    left[0] = Level({0, 2}, {-1, -1});
    left[1] = Level({1, 3}, {0, 1});
    right[0] = Level({0, 2}, {0, 1});
    right[1] = Level({0, 1}, {0, 0});
    right[2] = Level({0}, {-1});
    auto skeleton =
        std::make_shared<const FiberIndices>(std::move(left), std::move(right));

    TensorFibers<Scalar> G = rhs.evaluate(y, y.atFibers(skeleton), t);
    CHECK(G.skeleton() == skeleton);

    // Per-entry reference: c(i)^2 * ((L y)(i) + f(t) * s(i)), with L y formed
    // as a train once via zip.
    TensorTrain<Scalar> Ly =
        makeLaplacianOperator<Scalar>(sizes, spacings).zip(y, sizes, sizes);
    const auto expect = [&](const std::vector<Eigen::Index> &idx) {
        return c(idx) * c(idx) * (Ly(idx) + f(t) * s(idx));
    };
    const double tol = static_cast<double>(checkTol<Scalar>());

    // Slab 0: left index (i0), right index the (i1, i2) pair R0[col].
    for (Eigen::Index i = 0; i < 3; ++i) {
        for (Eigen::Index col = 0; col < 2; ++col) {
            const double e = static_cast<double>(
                expect({i, R0[col].first, R0[col].second}));
            CHECK(static_cast<double>(G.slab(0)(i, col)) ==
                  doctest::Approx(e).epsilon(tol));
        }
    }
    // Slab 1: left index L0[p], mode i1, right index R1[col].
    for (Eigen::Index p = 0; p < 2; ++p) {
        for (Eigen::Index i = 0; i < 4; ++i) {
            for (Eigen::Index col = 0; col < 2; ++col) {
                const double e =
                    static_cast<double>(expect({L0[p], i, R1[col]}));
                CHECK(static_cast<double>(G.slab(1)(p + 2 * i, col)) ==
                      doctest::Approx(e).epsilon(tol));
            }
        }
    }
    // Slab 2: left index the (i0, i1) pair L1[p], mode i2.
    for (Eigen::Index p = 0; p < 2; ++p) {
        for (Eigen::Index i = 0; i < 2; ++i) {
            const double e =
                static_cast<double>(expect({L1[p].first, L1[p].second, i}));
            CHECK(static_cast<double>(G.slab(2)(p + 2 * i, 0)) ==
                  doctest::Approx(e).epsilon(tol));
        }
    }
}

TEST_CASE_TEMPLATE(
    "Acoustic leapfrog propagates a Laplacian eigenmode exactly", Scalar,
    float, double) {
    // Source-free homogeneous medium (c = 1): a discrete Dirichlet eigenmode
    // p_0 stays rank 1 and obeys the scalar leapfrog recurrence
    // a_{n+1} = (2 + dt^2 * lambda) a_n - a_{n-1} exactly.
    const Eigen::Index n = 4;
    const Scalar h = Scalar(1);
    const std::vector<Eigen::Index> sizes{n, n};
    const std::vector<Scalar> spacings{h, h};
    const Scalar dt = Scalar(0.1);
    const int n_steps = 20;

    // First eigenvector of tridiag(1, -2, 1)/h^2: v_i = sin(pi (i+1)/(n+1)),
    // eigenvalue (2 cos(pi/(n+1)) - 2)/h^2.
    const Scalar pi = Scalar(3.14159265358979323846L);
    Eigen::VectorX<Scalar> v(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        v(i) = std::sin(pi * static_cast<Scalar>(i + 1) /
                        static_cast<Scalar>(n + 1));
    }
    const Scalar lambda_1d =
        (Scalar(2) * std::cos(pi / static_cast<Scalar>(n + 1)) - Scalar(2)) /
        (h * h);
    const Scalar lambda = Scalar(2) * lambda_1d; // sum over both dimensions

    auto p0 = makeRank1<Scalar>(v, v);
    Eigen::MatrixX<Scalar> dense0 = p0.toDense();

    const Eigen::VectorX<Scalar> ones = Eigen::VectorX<Scalar>::Ones(n);
    auto rhs = std::make_unique<AcousticRhs<Scalar>>(
        makeRank1<Scalar>(ones, ones), // c = 1
        makeRank1<Scalar>(ones, ones), // s (unused: f = 0)
        [](Scalar) { return Scalar(0); }, sizes, spacings);

    std::vector<TensorTrain<Scalar>> init;
    init.push_back(makeRank1<Scalar>(v, v)); // p_{-1} = p_0
    init.push_back(std::move(p0));
    Solver<Scalar> solver(std::move(init), std::move(rhs),
                          Scheme<Scalar>::leapfrogSecondOrder(), dt,
                          makeSelector<Scalar>(), Scalar(0),
                          checkTol<Scalar>());

    Scalar a_prev = Scalar(1);
    Scalar a = Scalar(1);
    for (int step = 0; step < n_steps; ++step) {
        solver.step();
        const Scalar a_next = (Scalar(2) + dt * dt * lambda) * a - a_prev;
        a_prev = a;
        a = a_next;
    }

    CHECK((solver.getState().toDense() - a * dense0).norm() <
          Scalar(100) * checkTol<Scalar>() * dense0.norm());
}

namespace {

// The per-axis Cerjan taper, mirroring makeCerjanMask.
template <typename Scalar>
Scalar taper1d(Eigen::Index i, Eigen::Index n, Eigen::Index width,
               Scalar strength) {
    const Eigen::Index dist = std::min(i, n - 1 - i);
    if (dist >= width) {
        return Scalar(1);
    }
    const Scalar x = strength * static_cast<Scalar>(width - dist);
    return std::exp(-x * x);
}

// F = 0: isolates the boundary condition in the solver.
template <typename Scalar> class ZeroRhs : public RhsBase<Scalar> {
  public:
    [[nodiscard]] TensorFibers<Scalar>
    evaluate(const TensorTrain<Scalar> &,
             const TensorFibers<Scalar> &state_fibers, Scalar) const override {
        return Scalar(0) * state_fibers;
    }
};

} // namespace

TEST_CASE_TEMPLATE("makeCerjanMask builds the separable rank-1 taper", Scalar,
                   float, double) {
    const std::vector<Eigen::Index> sizes{6, 5};
    const Eigen::Index width = 2;
    const Scalar strength = Scalar(0.4);

    TensorTrain<Scalar> mask = makeCerjanMask<Scalar>(sizes, width, strength);
    CHECK(mask.ranks() == std::vector<Eigen::Index>{1, 1, 1});

    const double tol = static_cast<double>(checkTol<Scalar>());
    for (Eigen::Index i0 = 0; i0 < 6; ++i0) {
        for (Eigen::Index i1 = 0; i1 < 5; ++i1) {
            const Scalar expected = taper1d<Scalar>(i0, 6, width, strength) *
                                    taper1d<Scalar>(i1, 5, width, strength);
            CHECK(static_cast<double>(
                      mask(std::vector<Eigen::Index>{i0, i1})) ==
                  doctest::Approx(static_cast<double>(expected)).epsilon(tol));
        }
    }

    // The deep interior is untouched: with width 2 on a 6-point axis, nodes
    // 2 and 3 carry taper 1.
    CHECK(static_cast<double>(mask(std::vector<Eigen::Index>{2, 2})) ==
          doctest::Approx(1.0));

    // Width 0 disables damping entirely.
    TensorTrain<Scalar> ones = makeCerjanMask<Scalar>(sizes, 0, strength);
    CHECK((ones.toDense() -
           Eigen::MatrixX<Scalar>::Ones(6 * 5, 1))
              .norm() < checkTol<Scalar>());
}

TEST_CASE_TEMPLATE("MaskBoundaryCondition damps the state once per step",
                   Scalar, float, double) {
    // Zero rhs + Cerjan mask: after n steps the state is D^{o n} ∘ y_0,
    // entry by entry. The mask is rank 1, so the Hadamard product preserves
    // the state's ranks and the cross rebuild stays exact.
    const std::vector<Eigen::Index> sizes{3, 4, 2};
    const Eigen::Index width = 1;
    const Scalar strength = Scalar(0.5);
    const int n_steps = 3;

    auto y0 = makeTrain<Scalar>(0);
    Eigen::MatrixX<Scalar> dense0 = y0.toDense();

    std::vector<TensorTrain<Scalar>> init;
    init.push_back(std::move(y0));
    Solver<Scalar> solver(
        std::move(init), std::make_unique<ZeroRhs<Scalar>>(),
        Scheme<Scalar>::forwardEuler(), Scalar(0.1), makeSelector<Scalar>(),
        Scalar(0), checkTol<Scalar>(), /*oversampling=*/0,
        std::make_unique<MaskBoundaryCondition<Scalar>>(
            makeCerjanMask<Scalar>(sizes, width, strength)));

    for (int n = 0; n < n_steps; ++n) {
        solver.step();
    }

    Eigen::MatrixX<Scalar> expected = dense0;
    for (Eigen::Index i0 = 0; i0 < 3; ++i0) {
        for (Eigen::Index i1 = 0; i1 < 4; ++i1) {
            for (Eigen::Index i2 = 0; i2 < 2; ++i2) {
                Scalar d = taper1d<Scalar>(i0, 3, width, strength) *
                           taper1d<Scalar>(i1, 4, width, strength) *
                           taper1d<Scalar>(i2, 2, width, strength);
                Scalar factor = Scalar(1);
                for (int n = 0; n < n_steps; ++n) {
                    factor *= d;
                }
                expected(i0 + 3 * i1 + 12 * i2, 0) *= factor;
            }
        }
    }

    CHECK((solver.getState().toDense() - expected).norm() <
          Scalar(100) * checkTol<Scalar>() * dense0.norm());
}
