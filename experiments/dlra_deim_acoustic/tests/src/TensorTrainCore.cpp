#include <doctest/doctest.h>

#include <Eigen/Core>

#include <TTCrossSolver/TensorTrainCore.h>

using MatSubset::Experiments::TensorTrainCore;

namespace {

// Builds a left unfolding (r0 * n) x r1 with deterministic, well-conditioned
// entries so the tests are reproducible across float/double.
template <typename Scalar>
Eigen::MatrixX<Scalar> makeUnfolding(Eigen::Index r0, Eigen::Index n,
                                     Eigen::Index r1) {
    Eigen::MatrixX<Scalar> A(r0 * n, r1);
    for (Eigen::Index i = 0; i < A.rows(); ++i) {
        for (Eigen::Index j = 0; j < A.cols(); ++j) {
            A(i, j) = static_cast<Scalar>(1 + (i * 7 + j * 3) % 11);
        }
    }
    return A;
}

// Tolerance for reconstruction / orthonormality checks, scaled per type.
template <typename Scalar> Scalar checkTol() {
    return std::is_same_v<Scalar, float> ? Scalar(1e-3) : Scalar(1e-9);
}

} // namespace

TEST_CASE_TEMPLATE("TensorTrainCore basic shape accessors", Scalar, float,
                   double) {
    TensorTrainCore<Scalar> core(2, 3, 4);
    CHECK(core.leftRank() == 2);
    CHECK(core.modeSize() == 3);
    CHECK(core.rightRank() == 4);
}

TEST_CASE_TEMPLATE("TensorTrainCore::modeSlice extracts G[:, i, :]", Scalar,
                   float, double) {
    const Eigen::Index r0 = 2, n = 3, r1 = 4;
    TensorTrainCore<Scalar> core(r0, n, r1);
    Eigen::MatrixX<Scalar> A = makeUnfolding<Scalar>(r0, n, r1);
    core.setLeftUnfolding(A);

    for (Eigen::Index i = 0; i < n; ++i) {
        Eigen::MatrixX<Scalar> slice = core.modeSlice(i);
        CHECK(slice.rows() == r0);
        CHECK(slice.cols() == r1);

        // Entry G[a, i, c] lives at row a + r0 * i, column c of the unfolding.
        for (Eigen::Index a = 0; a < r0; ++a) {
            for (Eigen::Index c = 0; c < r1; ++c) {
                CHECK(slice(a, c) == A(a + r0 * i, c));
            }
        }
    }
}

TEST_CASE_TEMPLATE(
    "TensorTrainCore::leftOrth (no R) reconstructs and orthonormalizes", Scalar,
    float, double) {
    const Eigen::Index r0 = 2, n = 3, r1 = 4;
    TensorTrainCore<Scalar> core(r0, n, r1);
    Eigen::MatrixX<Scalar> A = makeUnfolding<Scalar>(r0, n, r1);
    core.setLeftUnfolding(A);

    Eigen::MatrixX<Scalar> carry = core.leftOrth();
    const Eigen::MatrixX<Scalar> &Q = core.leftUnfolding();

    // Shapes: Q is (r0 * n) x r1, carry is r1 x r1.
    CHECK(Q.rows() == r0 * n);
    CHECK(Q.cols() == r1);
    CHECK(carry.rows() == r1);
    CHECK(carry.cols() == r1);

    // Q has orthonormal columns: Q^T Q = I.
    Eigen::MatrixX<Scalar> gram = Q.transpose() * Q;
    CHECK((gram - Eigen::MatrixX<Scalar>::Identity(r1, r1)).norm() <
          checkTol<Scalar>());

    // Round-trip: A = Q * carry (with column pivoting properly un-permuted).
    CHECK((A - Q * carry).norm() < checkTol<Scalar>() * A.norm());
}

TEST_CASE_TEMPLATE("TensorTrainCore::leftSvd truncates rank and reconstructs",
                   Scalar, float, double) {
    const Eigen::Index r0 = 2, n = 3, r1 = 4;
    TensorTrainCore<Scalar> core(r0, n, r1);

    // Build a rank-deficient unfolding: two independent columns, the rest are
    // linear combinations, so the numerical rank is 2.
    Eigen::MatrixX<Scalar> A(r0 * n, r1);
    Eigen::VectorX<Scalar> c0 = Eigen::VectorX<Scalar>::LinSpaced(r0 * n, 1, 6);
    Eigen::VectorX<Scalar> c1 =
        Eigen::VectorX<Scalar>::LinSpaced(r0 * n, -2, 3);
    A.col(0) = c0;
    A.col(1) = c1;
    A.col(2) = c0 + c1;
    A.col(3) = Scalar(2) * c0 - c1;
    core.setLeftUnfolding(A);

    Eigen::MatrixX<Scalar> carry =
        core.leftSvd(/*atol=*/Scalar(1e-6), /*rtol=*/Scalar(1e-6));
    const Eigen::MatrixX<Scalar> &U = core.leftUnfolding();

    // Rank should be truncated to 2.
    CHECK(U.cols() == 2);
    CHECK(carry.rows() == 2);
    CHECK(carry.cols() == r1);

    // U orthonormal columns.
    Eigen::MatrixX<Scalar> gram = U.transpose() * U;
    CHECK((gram - Eigen::MatrixX<Scalar>::Identity(U.cols(), U.cols())).norm() <
          checkTol<Scalar>());

    // Reconstruction within tolerance (R = I here so absorbed == A).
    CHECK((A - U * carry).norm() < Scalar(1e-4) * A.norm());

    // left_unfolding must now equal U and ranks must reflect the truncation.
    CHECK(core.leftRank() == r0);
    CHECK(core.rightRank() == 2);
    CHECK((core.leftUnfolding() - U).norm() < checkTol<Scalar>());
}

TEST_CASE_TEMPLATE("TensorTrainCore::rightSvd truncates rank and reconstructs",
                   Scalar, float, double) {
    const Eigen::Index r0 = 4, n = 3, r1 = 2;
    TensorTrainCore<Scalar> core(r0, n, r1);

    // Rank-deficient right unfolding r0 x (n*r1): make rows linearly dependent.
    Eigen::MatrixX<Scalar> A(r0 * n, r1);
    for (Eigen::Index j = 0; j < r1; ++j) {
        for (Eigen::Index i = 0; i < r0 * n; ++i) {
            // Row index mod 2 controls the pattern -> low row rank.
            A(i, j) = static_cast<Scalar>(((i % 2) + 1) * (j + 1));
        }
    }
    core.setLeftUnfolding(A);

    Eigen::MatrixX<Scalar> carry =
        core.rightSvd(/*atol=*/Scalar(1e-6), /*rtol=*/Scalar(1e-6));
    // The core's unfolding now holds V^T, stored as (rank * n) x r1; view it as
    // rank x (n * r1).
    Eigen::Map<const Eigen::MatrixX<Scalar>> Vt(
        core.leftUnfolding().data(), core.leftRank(),
        core.modeSize() * core.rightRank());

    // Vt has orthonormal rows.
    Eigen::MatrixX<Scalar> gram = Vt * Vt.transpose();
    CHECK(
        (gram - Eigen::MatrixX<Scalar>::Identity(Vt.rows(), Vt.rows())).norm() <
        checkTol<Scalar>());

    // carry (r0 x rank) * Vt (rank x (n*R.cols())) reconstructs the right
    // unfolding.
    Eigen::Map<Eigen::MatrixX<Scalar>> right_view(A.data(), r0, n * r1);
    Eigen::MatrixX<Scalar> right_unfolding = right_view;
    CHECK((right_unfolding - carry * Vt).norm() <
          Scalar(1e-4) * right_unfolding.norm());

    // left_unfolding must be updated: Vt reshaped to (rank * n) x r1,
    // and ranks must reflect the truncation.
    CHECK(core.leftRank() == Vt.rows());
    CHECK(core.rightRank() == r1);
    Eigen::Map<const Eigen::MatrixX<Scalar>> vt_as_left(Vt.data(),
                                                        Vt.rows() * n, r1);
    CHECK((core.leftUnfolding() - vt_as_left).norm() < checkTol<Scalar>());
}

TEST_CASE_TEMPLATE(
    "TensorTrainCore::rightOrth reconstructs and orthonormalizes", Scalar,
    float, double) {
    const Eigen::Index r0 = 2, n = 3, r1 = 2;
    TensorTrainCore<Scalar> core(r0, n, r1);
    Eigen::MatrixX<Scalar> A = makeUnfolding<Scalar>(r0, n, r1);
    core.setLeftUnfolding(A);

    Eigen::MatrixX<Scalar> carry = core.rightOrth();
    // The core's unfolding now holds the row-orthonormal factor, stored as
    // (r0 * n) x r1; view it as r0 x (n * r1).
    Eigen::Map<const Eigen::MatrixX<Scalar>> Qr(
        core.leftUnfolding().data(), core.leftRank(),
        core.modeSize() * core.rightRank());

    // Qr has orthonormal rows: Qr Qr^T = I.
    Eigen::MatrixX<Scalar> gram = Qr * Qr.transpose();
    CHECK(
        (gram - Eigen::MatrixX<Scalar>::Identity(Qr.rows(), Qr.rows())).norm() <
        checkTol<Scalar>());

    // Round-trip: the original right unfolding = carry * Qr.
    Eigen::Map<Eigen::MatrixX<Scalar>> right_view(A.data(), r0, n * r1);
    Eigen::MatrixX<Scalar> right_unfolding = right_view;
    CHECK((right_unfolding - carry * Qr).norm() <
          checkTol<Scalar>() * right_unfolding.norm());
}

TEST_CASE_TEMPLATE("TensorTrainCore::absorbRightFactor folds trailing carry",
                   Scalar, float, double) {
    const Eigen::Index r0 = 2, n = 3, r1 = 4;
    TensorTrainCore<Scalar> core(r0, n, r1);
    Eigen::MatrixX<Scalar> A = makeUnfolding<Scalar>(r0, n, r1);
    core.setLeftUnfolding(A);

    // A left-to-right sweep on the last core leaves a trailing carry r1 x r1f.
    const Eigen::Index r1f = 2;
    Eigen::MatrixX<Scalar> R(r1, r1f);
    R << 1, 2, 3, 4, 5, 6, 7, 8;

    core.absorbRightFactor(R);

    // right_rank updated, left_unfolding == A * R, shape (r0*n) x r1f.
    CHECK(core.rightRank() == r1f);
    CHECK(core.leftUnfolding().rows() == r0 * n);
    CHECK(core.leftUnfolding().cols() == r1f);
    CHECK((core.leftUnfolding() - A * R).norm() <
          checkTol<Scalar>() * (A * R).norm());
}

TEST_CASE_TEMPLATE("TensorTrainCore::absorbLeftFactor folds leading carry",
                   Scalar, float, double) {
    const Eigen::Index r0 = 3, n = 2, r1 = 2;
    TensorTrainCore<Scalar> core(r0, n, r1);
    Eigen::MatrixX<Scalar> A = makeUnfolding<Scalar>(r0, n, r1);
    core.setLeftUnfolding(A);

    // A right-to-left sweep on the first core leaves a leading carry r0f x r0.
    const Eigen::Index r0f = 2;
    Eigen::MatrixX<Scalar> R(r0f, r0);
    R << 1, 2, 3, 4, 5, 6;

    core.absorbLeftFactor(R);

    // left_rank updated to r0f, left_unfolding shape (r0f*n) x r1.
    CHECK(core.leftRank() == r0f);
    CHECK(core.leftUnfolding().rows() == r0f * n);
    CHECK(core.leftUnfolding().cols() == r1);

    // Compare against the reference absorption: view A as r0 x (n*r1),
    // left-multiply R, re-view as (r0f*n) x r1.
    Eigen::Map<Eigen::MatrixX<Scalar>> as_r0(A.data(), r0, n * r1);
    Eigen::MatrixX<Scalar> flat = R * as_r0;
    Eigen::Map<Eigen::MatrixX<Scalar>> expected(flat.data(), r0f * n, r1);
    CHECK((core.leftUnfolding() - expected).norm() <
          checkTol<Scalar>() * expected.norm());
}
