#include <doctest/doctest.h>

#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Core>

#include <MatSubset/DominantSelector.h>
#include <MatSubset/ForwardIterativeVolumeSamplingSelector.h>
#include <MatSubset/SelectorBase.h>

#include <TTCrossSolver/TensorFibers.h>
#include <TTCrossSolver/TensorTrain.h>
#include <TTCrossSolver/TensorTrainCore.h>

using MatSubset::Experiments::FiberIndices;
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

} // namespace

TEST_CASE_TEMPLATE("TensorTrain basic accessors", Scalar, float, double) {
    auto tt = makeTrain<Scalar>(3, 4, 2, 2, 3);
    CHECK(tt.order() == 3);

    auto modes = tt.modeSizes();
    CHECK(modes == std::vector<Eigen::Index>{3, 4, 2});

    auto r = tt.ranks();
    CHECK(r == std::vector<Eigen::Index>{1, 2, 3, 1});

    // toDense length is the product of mode sizes.
    CHECK(tt.toDense().rows() == 3 * 4 * 2);
    CHECK(tt.toDense().cols() == 1);
}

TEST_CASE_TEMPLATE("TensorTrain::operator() matches the dense tensor entry",
                   Scalar, float, double) {
    const Eigen::Index n0 = 3, n1 = 4, n2 = 2;
    auto tt = makeTrain<Scalar>(n0, n1, n2, 2, 3);
    Eigen::MatrixX<Scalar> dense = tt.toDense();

    // Evaluating at every multi-index reproduces the dense entry (first-mode-
    // fastest flattening, matching toDense).
    for (Eigen::Index i0 = 0; i0 < n0; ++i0) {
        for (Eigen::Index i1 = 0; i1 < n1; ++i1) {
            for (Eigen::Index i2 = 0; i2 < n2; ++i2) {
                const Scalar entry = tt(std::vector<Eigen::Index>{i0, i1, i2});
                const Scalar ref = dense(i0 + n0 * i1 + n0 * n1 * i2, 0);
                CHECK(std::abs(entry - ref) < checkTol<Scalar>());
            }
        }
    }
}

TEST_CASE_TEMPLATE("TensorTrain::operator() agrees with atFibers evaluation",
                   Scalar, float, double) {
    using Level = FiberIndices::Level;

    const Eigen::Index n0 = 3, n1 = 4, n2 = 2;
    auto tt = makeTrain<Scalar>(n0, n1, n2, 2, 2);

    // A nested skeleton (same layout as the atFibers test). Each slab entry is
    // a train evaluation at a fully determined multi-index, so operator() must
    // reproduce it.
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

    TensorFibers<Scalar> fibers = tt.atFibers(skeleton);

    // Slab 0: left index is just (i0), right index is the (i1, i2) pair R0[c].
    for (Eigen::Index i = 0; i < n0; ++i) {
        for (Eigen::Index c = 0; c < 2; ++c) {
            const Scalar entry =
                tt(std::vector<Eigen::Index>{i, R0[c].first, R0[c].second});
            CHECK(std::abs(fibers.slab(0)(i, c) - entry) < checkTol<Scalar>());
        }
    }

    // Slab 1: left index L0[p], mode i1, right index R1[c].
    for (Eigen::Index p = 0; p < 2; ++p) {
        for (Eigen::Index i = 0; i < n1; ++i) {
            for (Eigen::Index c = 0; c < 2; ++c) {
                const Scalar entry =
                    tt(std::vector<Eigen::Index>{L0[p], i, R1[c]});
                CHECK(std::abs(fibers.slab(1)(p + 2 * i, c) - entry) <
                      checkTol<Scalar>());
            }
        }
    }

    // Slab 2: left index is the (i0, i1) pair L1[p], mode i2, no right index.
    for (Eigen::Index p = 0; p < 2; ++p) {
        for (Eigen::Index i = 0; i < n2; ++i) {
            const Scalar entry =
                tt(std::vector<Eigen::Index>{L1[p].first, L1[p].second, i});
            CHECK(std::abs(fibers.slab(2)(p + 2 * i, 0) - entry) <
                  checkTol<Scalar>());
        }
    }
}

TEST_CASE_TEMPLATE("TensorTrain leftOrthogonalize preserves the tensor", Scalar,
                   float, double) {
    auto tt = makeTrain<Scalar>(3, 4, 2, 2, 3);
    Eigen::MatrixX<Scalar> before = tt.toDense();

    tt.leftOrthogonalize();
    Eigen::MatrixX<Scalar> after = tt.toDense();

    CHECK((before - after).norm() < checkTol<Scalar>() * before.norm());

    // Every core but the last has orthonormal left-unfolding columns.
    for (std::size_t k = 0; k + 1 < tt.order(); ++k) {
        const Eigen::MatrixX<Scalar> &U = tt.core(k).leftUnfolding();
        Eigen::MatrixX<Scalar> gram = U.transpose() * U;
        CHECK(
            (gram - Eigen::MatrixX<Scalar>::Identity(gram.rows(), gram.cols()))
                .norm() < checkTol<Scalar>());
    }
}

TEST_CASE_TEMPLATE("TensorTrain rightOrthogonalize preserves the tensor",
                   Scalar, float, double) {
    auto tt = makeTrain<Scalar>(3, 4, 2, 2, 3);
    Eigen::MatrixX<Scalar> before = tt.toDense();

    tt.rightOrthogonalize();
    Eigen::MatrixX<Scalar> after = tt.toDense();

    CHECK((before - after).norm() < checkTol<Scalar>() * before.norm());

    // Every core but the first has orthonormal right-unfolding rows:
    // viewing left_unfolding as r0 x (n * r1), rows are orthonormal.
    for (std::size_t k = 1; k < tt.order(); ++k) {
        const auto &core = tt.core(k);
        Eigen::Map<const Eigen::MatrixX<Scalar>> R(
            core.leftUnfolding().data(), core.leftRank(),
            core.modeSize() * core.rightRank());
        Eigen::MatrixX<Scalar> gram = R * R.transpose();
        CHECK(
            (gram - Eigen::MatrixX<Scalar>::Identity(gram.rows(), gram.cols()))
                .norm() < checkTol<Scalar>());
    }
}

TEST_CASE_TEMPLATE("TensorTrain repeated sweep is a no-op on the tensor",
                   Scalar, float, double) {
    auto tt = makeTrain<Scalar>(3, 4, 2, 2, 3);
    tt.leftOrthogonalize();
    Eigen::MatrixX<Scalar> once = tt.toDense();
    tt.leftOrthogonalize(); // should early-return, tensor unchanged
    Eigen::MatrixX<Scalar> twice = tt.toDense();
    CHECK((once - twice).norm() < checkTol<Scalar>() * once.norm());
}

TEST_CASE_TEMPLATE(
    "TensorTrain compress preserves a low-rank tensor and reduces "
    "the rank",
    Scalar, float, double) {
    // Construct a train whose bond ranks are inflated beyond the true rank by
    // padding with linearly dependent columns/rows, so SVD truncation should
    // recover the smaller rank while preserving the tensor.
    const Eigen::Index n0 = 4, n1 = 4, n2 = 4;
    const Eigen::Index true_r1 = 2, true_r2 = 2;

    // Build a genuine rank-(true_r) train, then it already has minimal ranks;
    // compressing at a loose tolerance must keep the tensor.
    auto tt = makeTrain<Scalar>(n0, n1, n2, true_r1, true_r2);
    Eigen::MatrixX<Scalar> before = tt.toDense();

    tt.compress(Scalar(0), checkTol<Scalar>());
    Eigen::MatrixX<Scalar> after = tt.toDense();

    CHECK((before - after).norm() < Scalar(1e-2) * before.norm());

    // Ranks should not exceed the originals.
    auto r = tt.ranks();
    CHECK(r.front() == 1);
    CHECK(r.back() == 1);
    CHECK(r[1] <= true_r1);
    CHECK(r[2] <= true_r2);
}

TEST_CASE_TEMPLATE("TensorTrain compress truncates a rank-inflated tensor",
                   Scalar, float, double) {
    // Core 1 carries a rank-2 bond but its unfolding is built to be rank 1 in
    // the bond direction; compression should collapse it.
    std::vector<TensorTrainCore<Scalar>> cores;

    // First core: 1 x 3 x 2, but both columns proportional -> effective r1 = 1.
    Eigen::MatrixX<Scalar> g0(3, 2);
    g0.col(0) << Scalar(1), Scalar(2), Scalar(3);
    g0.col(1) = Scalar(10) * g0.col(0); // dependent -> rank 1
    cores.emplace_back(g0, /*n=*/3);

    // Middle core 2 x 2 x 1.
    cores.emplace_back(makeUnfolding<Scalar>(2, 2, 1, 4), /*n=*/2);

    TensorTrain<Scalar> tt(std::move(cores));
    Eigen::MatrixX<Scalar> before = tt.toDense();

    tt.compress(Scalar(0), checkTol<Scalar>());
    Eigen::MatrixX<Scalar> after = tt.toDense();

    CHECK((before - after).norm() < Scalar(1e-2) * before.norm());
    CHECK(tt.ranks()[1] == 1); // the inflated bond collapsed to 1
}

TEST_CASE_TEMPLATE("TensorTrain::selectIndices truncates, selects a skeleton "
                   "and returns self-consistent fibers",
                   Scalar, float, double) {
    auto tt = makeTrain<Scalar>(3, 4, 2, 2, 3);
    tt.leftOrthogonalize(); // selectIndices assumes a left-orthogonal train
    Eigen::MatrixX<Scalar> dense = tt.toDense();

    std::unique_ptr<MatSubset::SelectorBase<Scalar>> selector =
        std::make_unique<MatSubset::DominantSelector<Scalar>>(Scalar(1));

    TensorFibers<Scalar> fibers = tt.selectIndices(selector, /*atol=*/Scalar(0),
                                                   /*rtol=*/checkTol<Scalar>());

    // The TT-SVD half of the sweep preserves the tensor (bond 1 truncates from
    // 3 to the maximal possible rank 2 without loss).
    CHECK((tt.toDense() - dense).norm() <
          Scalar(100) * checkTol<Scalar>() * dense.norm());

    // Skeleton: one selection level per interior bond, sized by the truncated
    // bond ranks (no oversampling).
    const auto &skeleton = *fibers.skeleton();
    auto r = tt.ranks();
    REQUIRE(skeleton.order() == tt.order());
    for (std::size_t k = 0; k + 1 < tt.order(); ++k) {
        CHECK(static_cast<Eigen::Index>(skeleton.leftLevel(k).size()) ==
              r[k + 1]);
        CHECK(static_cast<Eigen::Index>(skeleton.rightLevel(k).size()) ==
              r[k + 1]);
    }

    // Slabs have the fiber shapes (leftFiberCount(k) * n_k) x rightFiberCount.
    for (std::size_t k = 0; k < tt.order(); ++k) {
        CHECK(fibers.slab(k).rows() ==
              static_cast<Eigen::Index>(skeleton.leftFiberCount(k)) *
                  tt.core(k).modeSize());
        CHECK(fibers.slab(k).cols() ==
              static_cast<Eigen::Index>(skeleton.rightFiberCount(k)));
    }

    // atFibers re-evaluates the (mutated) train on the selected skeleton and
    // reproduces selectIndices' slabs exactly.
    {
        TensorFibers<Scalar> reeval = tt.atFibers(fibers.skeleton());
        for (std::size_t k = 0; k < tt.order(); ++k) {
            CHECK((reeval.slab(k) - fibers.slab(k)).norm() <
                  Scalar(100) * checkTol<Scalar>());
        }
    }

    // On exit the train is left-orthogonal again (sweep 2 orthonormalizes each
    // core before the selector runs on it): cores 0..d-2 have orthonormal left
    // unfoldings.
    for (std::size_t k = 0; k + 1 < tt.order(); ++k) {
        const Eigen::MatrixX<Scalar> &U = tt.core(k).leftUnfolding();
        Eigen::MatrixX<Scalar> gram = U.transpose() * U;
        CHECK(
            (gram - Eigen::MatrixX<Scalar>::Identity(gram.rows(), gram.cols()))
                .norm() < Scalar(100) * checkTol<Scalar>());
    }

    // Self-consistency: reconstructing a train from the returned fibers
    // reproduces the tensor.
    TensorTrain<Scalar> rebuilt(fibers);
    CHECK((rebuilt.toDense() - dense).norm() <
          Scalar(100) * checkTol<Scalar>() * dense.norm());
}

TEST_CASE_TEMPLATE("TensorTrain(fibers) reconstructs a low-rank tensor in "
                   "left-orthogonal form",
                   Scalar, float, double) {
    using Level = FiberIndices::Level;

    // Ground truth: rank-(1,2,2,1) train with modes (3, 4, 2).
    const Eigen::Index n0 = 3, n1 = 4, n2 = 2;
    auto truth = makeTrain<Scalar>(n0, n1, n2, 2, 2);
    Eigen::MatrixX<Scalar> dense = truth.toDense();
    auto T = [&](Eigen::Index i0, Eigen::Index i1, Eigen::Index i2) {
        return dense(i0 + n0 * i1 + n0 * n1 * i2, 0);
    };

    // Nested left multi-indices: L0 = {(0), (2)},
    // L1 = {(0, 1), (2, 3)} (each node = parent's tuple + appended mode).
    std::vector<Eigen::Index> L0 = {0, 2};
    std::vector<std::pair<Eigen::Index, Eigen::Index>> L1 = {{0, 1}, {2, 3}};

    // Right multi-index sets: bond 0 gets (i1, i2) pairs, bond 1 gets (i2).
    std::vector<std::pair<Eigen::Index, Eigen::Index>> R0 = {{0, 0}, {2, 1}};
    std::vector<Eigen::Index> R1 = {0, 1};

    std::vector<Level> left(3), right(3);
    left[0] = Level({0, 2}, {-1, -1});
    left[1] = Level({1, 3}, {0, 1});
    // left[2] stays empty: no left set at the last bond.
    right[0] = Level({0, 2}, {0, 1});
    right[1] = Level({0, 1}, {0, 0}); // parents point to the root at right[2]
    right[2] = Level({0}, {-1});
    auto skeleton =
        std::make_shared<const FiberIndices>(std::move(left), std::move(right));

    // Sample the slabs from the dense tensor on the skeleton's fibers.
    std::vector<Eigen::MatrixX<Scalar>> slabs(3);
    slabs[0].resize(n0, 2);
    for (Eigen::Index i = 0; i < n0; ++i) {
        for (Eigen::Index c = 0; c < 2; ++c) {
            slabs[0](i, c) = T(i, R0[c].first, R0[c].second);
        }
    }
    slabs[1].resize(2 * n1, 2);
    for (Eigen::Index p = 0; p < 2; ++p) {
        for (Eigen::Index i = 0; i < n1; ++i) {
            for (Eigen::Index c = 0; c < 2; ++c) {
                slabs[1](p + 2 * i, c) = T(L0[p], i, R1[c]);
            }
        }
    }
    slabs[2].resize(2 * n2, 1);
    for (Eigen::Index p = 0; p < 2; ++p) {
        for (Eigen::Index i = 0; i < n2; ++i) {
            slabs[2](p + 2 * i, 0) = T(L1[p].first, L1[p].second, i);
        }
    }

    TensorFibers<Scalar> fibers(std::move(slabs), skeleton);
    TensorTrain<Scalar> tt(fibers);

    // Shapes and boundary ranks.
    CHECK(tt.order() == 3);
    auto r = tt.ranks();
    CHECK(r == std::vector<Eigen::Index>{1, 2, 2, 1});

    // Left-orthogonal by construction: cores 0..d-2 have orthonormal left
    // unfoldings.
    for (std::size_t k = 0; k + 1 < tt.order(); ++k) {
        const Eigen::MatrixX<Scalar> &U = tt.core(k).leftUnfolding();
        Eigen::MatrixX<Scalar> gram = U.transpose() * U;
        CHECK(
            (gram - Eigen::MatrixX<Scalar>::Identity(gram.rows(), gram.cols()))
                .norm() < checkTol<Scalar>());
    }

    // Cross interpolation is exact for a tensor of matching TT-rank.
    CHECK((tt.toDense() - dense).norm() <
          Scalar(100) * checkTol<Scalar>() * dense.norm());
}

TEST_CASE_TEMPLATE("TensorTrain::atFibers evaluates the train on a skeleton",
                   Scalar, float, double) {
    using Level = FiberIndices::Level;

    const Eigen::Index n0 = 3, n1 = 4, n2 = 2;
    auto tt = makeTrain<Scalar>(n0, n1, n2, 2, 2);
    Eigen::MatrixX<Scalar> dense = tt.toDense();
    auto T = [&](Eigen::Index i0, Eigen::Index i1, Eigen::Index i2) {
        return dense(i0 + n0 * i1 + n0 * n1 * i2, 0);
    };

    // A nested skeleton (same layout as the reconstruction test above).
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

    TensorFibers<Scalar> fibers = tt.atFibers(skeleton);

    // atFibers does not mutate: contracting the train still gives `dense`.
    CHECK((tt.toDense() - dense).norm() < checkTol<Scalar>() * dense.norm());
    CHECK(fibers.skeleton() == skeleton); // shares the passed-in skeleton

    // Slabs match the tensor entries at the skeleton's fibers, sampled by hand.
    Eigen::MatrixX<Scalar> slab0(n0, 2);
    for (Eigen::Index i = 0; i < n0; ++i) {
        for (Eigen::Index c = 0; c < 2; ++c) {
            slab0(i, c) = T(i, R0[c].first, R0[c].second);
        }
    }
    CHECK((fibers.slab(0) - slab0).norm() < checkTol<Scalar>());

    Eigen::MatrixX<Scalar> slab1(2 * n1, 2);
    for (Eigen::Index p = 0; p < 2; ++p) {
        for (Eigen::Index i = 0; i < n1; ++i) {
            for (Eigen::Index c = 0; c < 2; ++c) {
                slab1(p + 2 * i, c) = T(L0[p], i, R1[c]);
            }
        }
    }
    CHECK((fibers.slab(1) - slab1).norm() < checkTol<Scalar>());

    Eigen::MatrixX<Scalar> slab2(2 * n2, 1);
    for (Eigen::Index p = 0; p < 2; ++p) {
        for (Eigen::Index i = 0; i < n2; ++i) {
            slab2(p + 2 * i, 0) = T(L1[p].first, L1[p].second, i);
        }
    }
    CHECK((fibers.slab(2) - slab2).norm() < checkTol<Scalar>());
}

TEST_CASE_TEMPLATE("TensorTrainCore::zip contracts one operator/tensor core",
                   Scalar, float, double) {
    // Single-core operator (rank 1): its zip with a rank-1 tensor core is just
    // the m x n matrix times the length-n vector.
    const Eigen::Index m = 3, n = 4;
    // Operator core 1 x (m*n) x 1, mode folded as out*n + in.
    Eigen::MatrixX<Scalar> a(m * n, 1);
    Eigen::MatrixX<Scalar> M(m, n);
    for (Eigen::Index out = 0; out < m; ++out) {
        for (Eigen::Index in = 0; in < n; ++in) {
            const Scalar val = static_cast<Scalar>(1 + (out * 5 + in * 3) % 7);
            M(out, in) = val;
            a(out * n + in, 0) = val;
        }
    }
    TensorTrainCore<Scalar> op(a, m * n);

    // Tensor core 1 x n x 1 = a length-n vector.
    Eigen::MatrixX<Scalar> b(n, 1);
    Eigen::VectorX<Scalar> v(n);
    for (Eigen::Index in = 0; in < n; ++in) {
        b(in, 0) = static_cast<Scalar>(1 + (in * 2 + 1) % 5);
        v(in) = b(in, 0);
    }
    TensorTrainCore<Scalar> vec(b, n);

    TensorTrainCore<Scalar> res = op.zip(vec, /*out_size=*/m, /*in_size=*/n);
    CHECK(res.leftRank() == 1);
    CHECK(res.rightRank() == 1);
    CHECK(res.modeSize() == m);

    Eigen::VectorX<Scalar> expected = M * v;
    for (Eigen::Index out = 0; out < m; ++out) {
        CHECK(std::abs(res.modeSlice(out)(0, 0) - expected(out)) <
              checkTol<Scalar>());
    }
}

TEST_CASE_TEMPLATE("TensorTrain::zip applies a TT operator to a TT tensor",
                   Scalar, float, double) {
    // A 2-core operator with output sizes (m0, m1) and input sizes (n0, n1),
    // and a matching 2-core tensor with mode sizes (n0, n1).
    const Eigen::Index m0 = 2, n0 = 3, m1 = 3, n1 = 2;
    const Eigen::Index opr = 2; // operator middle bond rank
    const Eigen::Index tsr = 2; // tensor middle bond rank

    std::vector<TensorTrainCore<Scalar>> op_cores;
    op_cores.emplace_back(makeUnfolding<Scalar>(1, m0 * n0, opr, 7), m0 * n0);
    op_cores.emplace_back(makeUnfolding<Scalar>(opr, m1 * n1, 1, 8), m1 * n1);
    TensorTrain<Scalar> op(std::move(op_cores));

    std::vector<TensorTrainCore<Scalar>> vec_cores;
    vec_cores.emplace_back(makeUnfolding<Scalar>(1, n0, tsr, 9), n0);
    vec_cores.emplace_back(makeUnfolding<Scalar>(tsr, n1, 1, 10), n1);
    TensorTrain<Scalar> vec(std::move(vec_cores));

    TensorTrain<Scalar> res =
        op.zip(vec, /*out_sizes=*/{m0, m1}, /*in_sizes=*/{n0, n1});

    // Shapes: mode sizes are the output sizes, bond ranks are the products.
    CHECK(res.modeSizes() == std::vector<Eigen::Index>{m0, m1});
    CHECK(res.ranks() == std::vector<Eigen::Index>{1, opr * tsr, 1});

    // Ground truth: build the operator's dense matrix M (row = output multi-
    // index, col = input multi-index) and vector, and compare M*v to the
    // zipped train contracted to dense. All multi-indices are first-mode-fast.
    Eigen::MatrixX<Scalar> op_dense = op.toDense(); // length (m0*n0)*(m1*n1)
    Eigen::MatrixX<Scalar> v = vec.toDense();       // length n0*n1

    Eigen::MatrixX<Scalar> M(m0 * m1, n0 * n1);
    for (Eigen::Index o0 = 0; o0 < m0; ++o0) {
        for (Eigen::Index o1 = 0; o1 < m1; ++o1) {
            for (Eigen::Index i0 = 0; i0 < n0; ++i0) {
                for (Eigen::Index i1 = 0; i1 < n1; ++i1) {
                    const Eigen::Index j0 = o0 * n0 + i0; // core-0 folded mode
                    const Eigen::Index j1 = o1 * n1 + i1; // core-1 folded mode
                    const Eigen::Index flat = j0 + (m0 * n0) * j1;
                    M(o0 + m0 * o1, i0 + n0 * i1) = op_dense(flat, 0);
                }
            }
        }
    }

    Eigen::MatrixX<Scalar> expected = M * v;
    Eigen::MatrixX<Scalar> got = res.toDense();
    REQUIRE(got.rows() == expected.rows());
    CHECK((got - expected).norm() <
          Scalar(100) * checkTol<Scalar>() * expected.norm());
}

TEST_CASE_TEMPLATE(
    "TensorTrain(fibers) is stable on an oversampled skeleton of a localized "
    "tensor",
    Scalar, float, double) {
    // Regression test: a sharply localized field whose fibers vanish at most
    // grid points, sampled on a skeleton much wider than its rank. The naive
    // rebuild (full QR of the rank-deficient slabs) hands the pseudo-inverse
    // arbitrary null-space directions with near-zero rows and amplifies noise
    // catastrophically; the truncating rebuild must stay exact and collapse
    // the ranks to the content rank.
    const Eigen::Index n = 32;
    // Width policy: rank + 12 - far wider than the content rank of 2.
    const auto num_samples = [](Eigen::Index rank, Eigen::Index) {
        return rank + 12;
    };

    // Rank-1 Gaussian bumps, two cells wide, centered off-node.
    const auto gaussian = [n](Scalar center, Scalar sigma) {
        Eigen::MatrixX<Scalar> g(n, 1);
        for (Eigen::Index i = 0; i < n; ++i) {
            const Scalar d = static_cast<Scalar>(i) - center;
            g(i, 0) = std::exp(-d * d / (Scalar(2) * sigma * sigma));
        }
        return g;
    };
    const auto bump = [&](Scalar center, Scalar sigma) {
        std::vector<TensorTrainCore<Scalar>> cores;
        cores.emplace_back(gaussian(center, sigma), n);
        cores.emplace_back(gaussian(center, sigma), n);
        cores.emplace_back(gaussian(center, sigma), n);
        return TensorTrain<Scalar>(std::move(cores));
    };

    TensorTrain<Scalar> y = bump(Scalar(15.5), Scalar(2));
    TensorTrain<Scalar> z = bump(Scalar(19.5), Scalar(3));
    Eigen::MatrixX<Scalar> target = y.toDense() + z.toDense();

    // Skeleton from y alone (as the solver does), then fibers of y + z.
    std::unique_ptr<MatSubset::SelectorBase<Scalar>> selector =
        std::make_unique<
            MatSubset::ForwardIterativeVolumeSamplingSelector<Scalar>>(12345);
    y.leftOrthogonalize();
    TensorFibers<Scalar> fy =
        y.selectIndices(selector, Scalar(0), checkTol<Scalar>(), num_samples);
    TensorFibers<Scalar> combo = fy + Scalar(1) * z.atFibers(fy.skeleton());
    TensorTrain<Scalar> rebuilt(combo);

    CHECK((rebuilt.toDense() - target).norm() <
          Scalar(100) * checkTol<Scalar>() * target.norm());

    // The rebuild must not inflate the ranks to the skeleton width: the
    // content rank of y + z is 2 per bond.
    for (const Eigen::Index r : rebuilt.ranks()) {
        CHECK(r <= 3);
    }
}

TEST_CASE_TEMPLATE("TensorTrain operator+, scalar multiply and "
                   "hadamardProduct match dense arithmetic",
                   Scalar, float, double) {
    auto a = makeTrain<Scalar>(3, 4, 2, 2, 2);
    std::vector<TensorTrainCore<Scalar>> b_cores;
    b_cores.emplace_back(makeUnfolding<Scalar>(1, 3, 3, 7), 3);
    b_cores.emplace_back(makeUnfolding<Scalar>(3, 4, 2, 8), 4);
    b_cores.emplace_back(makeUnfolding<Scalar>(2, 2, 1, 9), 2);
    TensorTrain<Scalar> b(std::move(b_cores));

    Eigen::MatrixX<Scalar> da = a.toDense();
    Eigen::MatrixX<Scalar> db = b.toDense();
    const Scalar tol = checkTol<Scalar>();

    // Sum: bond ranks add, entries add.
    TensorTrain<Scalar> sum = a + b;
    CHECK(sum.ranks() == std::vector<Eigen::Index>{1, 5, 4, 1});
    CHECK((sum.toDense() - (da + db)).norm() <
          Scalar(100) * tol * (da + db).norm());

    // Scalar multiply.
    TensorTrain<Scalar> scaled = Scalar(-2.5) * a;
    CHECK((scaled.toDense() + Scalar(2.5) * da).norm() <
          Scalar(100) * tol * da.norm());

    // Hadamard: bond ranks multiply, entries multiply.
    TensorTrain<Scalar> had = hadamardProduct(a, b);
    CHECK(had.ranks() == std::vector<Eigen::Index>{1, 6, 4, 1});
    Eigen::MatrixX<Scalar> expected = da.cwiseProduct(db);
    CHECK((had.toDense() - expected).norm() <
          Scalar(100) * tol * expected.norm());

    // Single-core trains: the boundary blocks share the singleton bond.
    std::vector<TensorTrainCore<Scalar>> ca, cb;
    ca.emplace_back(makeUnfolding<Scalar>(1, 5, 1, 3), 5);
    cb.emplace_back(makeUnfolding<Scalar>(1, 5, 1, 4), 5);
    TensorTrain<Scalar> a1(std::move(ca)), b1(std::move(cb));
    CHECK((TensorTrain<Scalar>(a1 + b1).toDense() -
           (a1.toDense() + b1.toDense()))
              .norm() < Scalar(100) * tol * a1.toDense().norm());
}
