#include <memory>

#include <doctest/doctest.h>

#include <MatSubset/DerandomizedVolumeSelector.h>

#include "CheckBounds.h"
#include "CheckSubset.h"

TEST_CASE_TEMPLATE("derandomized volume", Scalar, float, double) {

    std::unique_ptr<MatSubset::DerandomizedVolumeSelector<Scalar>> selector =
        std::make_unique<MatSubset::DerandomizedVolumeSelector<Scalar>>();

    // Algorithm name
    CHECK(selector->getAlgorithmName() == "derandomized volume");

    // Check that returned vector is subset of column indices
    check_subset(selector.get(), 3, 5);

    // Check bounds
    check_bounds(selector.get(), 3, 5);

    // Regression: m=2, n=3, k=2 (n-k=1) crashed with an Eigen Block assertion
    // because len=0 (no active eigenvalues) led to g.middleRows() on a 0-row matrix.
    SUBCASE("2x3 matrix, k=2 (n-k=1 edge case)") {
        Eigen::MatrixX<Scalar> X(2, 3);
        X << 1, 2, 3, 4, 5, 7;
        std::vector<Eigen::Index> idx = selector->selectSubset(X, 2);
        CHECK(idx.size() == 2);
        std::set<Eigen::Index> idx_set(idx.begin(), idx.end());
        CHECK(idx_set.size() == 2);
        CHECK(*idx_set.begin() >= 0);
        CHECK(*idx_set.rbegin() <= 2);
    }
}
