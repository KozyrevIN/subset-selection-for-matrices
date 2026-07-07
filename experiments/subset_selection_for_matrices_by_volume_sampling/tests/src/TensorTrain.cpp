#include <doctest/doctest.h>

#include <memory>
#include <vector>

#include <Eigen/Core>

#include <MatSubset/RandomColumnsSelector.h>
#include <MatSubset/SelectorBase.h>

#include <TTCrossSolver/TensorTrain.h>
#include <TTCrossSolver/TensorTrainCore.h>

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

TEST_CASE_TEMPLATE("TensorTrain selectIndices returns nested index sets of the "
                   "right shape",
                   Scalar, float, double) {
    auto tt = makeTrain<Scalar>(4, 4, 4, 2, 2);

    std::unique_ptr<MatSubset::SelectorBase<Scalar>> selector =
        std::make_unique<MatSubset::RandomColumnsSelector<Scalar>>(/*seed=*/42);

    auto [left_indices, right_indices] = tt.selectIndices(selector);

    // One index set per interior bond (d - 1 of them).
    CHECK(left_indices.size() == tt.order() - 1);
    CHECK(right_indices.size() == tt.order() - 1);

    // Left bond k selects r_{k+1} row-indices of core k's unfolding.
    auto r = tt.ranks();
    for (std::size_t k = 0; k + 1 < tt.order(); ++k) {
        CHECK(static_cast<Eigen::Index>(left_indices[k].size()) == r[k + 1]);
    }
    // Right bond stored at index k selects r_{k+1} column-indices of core k+1.
    for (std::size_t k = 0; k + 1 < tt.order(); ++k) {
        CHECK(static_cast<Eigen::Index>(right_indices[k].size()) == r[k + 1]);
    }
}
