#include <memory>

#include <doctest/doctest.h>

#include <MatSubset/QdeimSelector.h>

#include "CheckBounds.h"
#include "CheckSubset.h"

TEST_CASE_TEMPLATE("qdeim", Scalar, float, double) {

    std::unique_ptr<MatSubset::QdeimSelector<Scalar>> selector =
        std::make_unique<MatSubset::QdeimSelector<Scalar>>();

    // Algorithm name
    CHECK(selector->getAlgorithmName() == "qdeim");

    // Check that returned vector is subset of column indices. Q-DEIM returns
    // one interpolation point per basis vector, so k = m = 3 is the only valid
    // k.
    check_subset(selector.get(), 3, 3);

    // Check bounds
    check_bounds(selector.get(), 3, 3);

    SUBCASE("First pivot is the column of largest norm") {
        // Column-pivoted QR always takes the largest-norm column first, on the
        // orthonormal row-space basis rather than on X itself. With an
        // orthogonal X the two coincide, which pins the pivot deterministically.
        const Eigen::Index m = 3;
        const Eigen::Index n = 5;

        Eigen::MatrixX<Scalar> X = Eigen::MatrixX<Scalar>::Zero(m, n);
        X(0, 0) = 1;
        X(1, 1) = 1;
        X(2, 4) = 1;

        std::vector<Eigen::Index> indices = selector->selectSubset(X, m);
        std::sort(indices.begin(), indices.end());

        CHECK(indices[0] == 0);
        CHECK(indices[1] == 1);
        CHECK(indices[2] == 4);
    }

    SUBCASE("Selected submatrix is nonsingular") {
        const Eigen::Index m = 3;
        const Eigen::Index n = 6;

        Eigen::MatrixX<Scalar> X(m, n);
        // clang-format off
        X <<  1,  2,  3,  4,  5,  6,
              0,  6,  7,  8,  9, 10,
              0,  0, 10, 11, 12, 13;
        // clang-format on

        std::vector<Eigen::Index> indices = selector->selectSubset(X, m);
        Eigen::MatrixX<Scalar> submatrix = X(Eigen::all, indices);

        Eigen::FullPivLU<Eigen::MatrixX<Scalar>> lu(submatrix);
        CHECK(lu.rank() == m);
    }
}
