#include <memory>

#include <doctest/doctest.h>

#include <MatSubset/RectMaxvolSelector.h>

#include "CheckBounds.h"
#include "CheckSubset.h"

TEST_CASE_TEMPLATE("rect-maxvol", Scalar, float, double) {

    std::unique_ptr<MatSubset::RectMaxvolSelector<Scalar>> selector =
        std::make_unique<MatSubset::RectMaxvolSelector<Scalar>>(1.01);

    // Algorithm name
    CHECK(selector->getAlgorithmName() == "rect-maxvol");

    // Check that returned vector is subset of column indices
    check_subset(selector.get(), 3, 5);

    // Check bounds
    check_subset(selector.get(), 3, 5);
}