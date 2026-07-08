#include <doctest/doctest.h>

#include <memory>
#include <vector>

#include <Eigen/Core>

#include <MatSubset/DominantSelector.h>
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
        CHECK((gram - Eigen::MatrixX<Scalar>::Identity(gram.rows(), gram.cols()))
                  .norm() < checkTol<Scalar>());
    }
}

TEST_CASE_TEMPLATE("TensorTrain rightOrthogonalize preserves the tensor", Scalar,
                   float, double) {
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
        CHECK((gram - Eigen::MatrixX<Scalar>::Identity(gram.rows(), gram.cols()))
                  .norm() < checkTol<Scalar>());
    }
}

TEST_CASE_TEMPLATE("TensorTrain repeated sweep is a no-op on the tensor", Scalar,
                   float, double) {
    auto tt = makeTrain<Scalar>(3, 4, 2, 2, 3);
    tt.leftOrthogonalize();
    Eigen::MatrixX<Scalar> once = tt.toDense();
    tt.leftOrthogonalize(); // should early-return, tensor unchanged
    Eigen::MatrixX<Scalar> twice = tt.toDense();
    CHECK((once - twice).norm() < checkTol<Scalar>() * once.norm());
}

TEST_CASE_TEMPLATE("TensorTrain compress preserves a low-rank tensor and reduces "
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

    TensorFibers<Scalar> fibers =
        tt.selectIndices(selector, /*atol=*/Scalar(0),
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

    // On exit the train is left-orthogonal again (sweep 2 orthonormalizes each
    // core before the selector runs on it): cores 0..d-2 have orthonormal left
    // unfoldings.
    for (std::size_t k = 0; k + 1 < tt.order(); ++k) {
        const Eigen::MatrixX<Scalar> &U = tt.core(k).leftUnfolding();
        Eigen::MatrixX<Scalar> gram = U.transpose() * U;
        CHECK((gram - Eigen::MatrixX<Scalar>::Identity(gram.rows(),
                                                       gram.cols()))
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
    auto skeleton = std::make_shared<const FiberIndices>(std::move(left),
                                                         std::move(right));

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
        CHECK((gram - Eigen::MatrixX<Scalar>::Identity(gram.rows(),
                                                       gram.cols()))
                  .norm() < checkTol<Scalar>());
    }

    // Cross interpolation is exact for a tensor of matching TT-rank.
    CHECK((tt.toDense() - dense).norm() <
          Scalar(100) * checkTol<Scalar>() * dense.norm());
}
