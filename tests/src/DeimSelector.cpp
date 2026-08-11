#include <memory>

#include <doctest/doctest.h>

#include <MatSubset/DeimSelector.h>

#include "CheckBounds.h"
#include "CheckSubset.h"

TEST_CASE_TEMPLATE("deim", Scalar, float, double) {

    std::unique_ptr<MatSubset::DeimSelector<Scalar>> selector =
        std::make_unique<MatSubset::DeimSelector<Scalar>>();

    // Algorithm name
    CHECK(selector->getAlgorithmName() == "deim");

    // Check that returned vector is subset of column indices. DEIM returns one
    // interpolation point per basis vector, so k = m = 3 is the only valid k.
    check_subset(selector.get(), 3, 3);

    // Check bounds
    check_bounds(selector.get(), 3, 3);

    SUBCASE("Interpolation is exact on the selected rows") {
        // With m = n the basis is a full orthonormal set, so DEIM must pick
        // every index exactly once and interpolation is exact everywhere.
        const Eigen::Index m = 4;
        Eigen::MatrixX<Scalar> X = Eigen::MatrixX<Scalar>::Identity(m, m);

        std::vector<Eigen::Index> indices = selector->selectSubset(X, m);
        std::sort(indices.begin(), indices.end());
        for (Eigen::Index i = 0; i < m; ++i) {
            CHECK(indices[static_cast<size_t>(i)] == i);
        }
    }

    SUBCASE("Selected submatrix is nonsingular") {
        // The DEIM guarantee rests on P^T U being invertible; a rank-deficient
        // pick would make the error bound meaningless.
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
