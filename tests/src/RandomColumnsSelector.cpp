#include <memory>

#include <doctest/doctest.h>

#include <MatSubset/RandomColumnsSelector.h>

#include "CheckBounds.h"
#include "CheckSubset.h"

TEST_CASE_TEMPLATE("random columns", Scalar, float, double) {

    std::unique_ptr<MatSubset::RandomColumnsSelector<Scalar>> selector =
        std::make_unique<MatSubset::RandomColumnsSelector<Scalar>>();

    // Algorithm name
    CHECK(selector->getAlgorithmName() == "random columns");

    // Check that returned vector is subset of column indices
    check_subset(selector.get(), 3, 5);

    // Check bounds
    check_subset(selector.get(), 3, 5);
}