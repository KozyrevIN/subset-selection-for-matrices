#include <memory>

#include <doctest/doctest.h>

#include <MatSubset/FrobeniusRemovalSelector.h>

#include "CheckBounds.h"
#include "CheckSubset.h"

TEST_CASE_TEMPLATE("frobenius removal", Scalar, float, double) {

    std::unique_ptr<MatSubset::FrobeniusRemovalSelector<Scalar>> selector =
        std::make_unique<MatSubset::FrobeniusRemovalSelector<Scalar>>();

    // Algorithm name
    CHECK(selector->getAlgorithmName() == "frobenius removal");

    // Check that returned vector is subset of column indices
    check_subset(selector.get(), 3, 5);

    // Check bounds
    check_subset(selector.get(), 3, 5);
}