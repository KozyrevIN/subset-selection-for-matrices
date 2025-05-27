#include <memory>

#include <doctest/doctest.h>

#include <MatSubset/VolumeRemovalSelector.h>

#include "CheckBounds.h"
#include "CheckSubset.h"

TEST_CASE_TEMPLATE("volume removal", Scalar, float, double) {

    std::unique_ptr<MatSubset::VolumeRemovalSelector<Scalar>> selector =
        std::make_unique<MatSubset::VolumeRemovalSelector<Scalar>>();

    // Algorithm name
    CHECK(selector->getAlgorithmName() == "volume removal");

    // Check that returned vector is subset of column indices
    check_subset(selector.get(), 3, 5);

    // Check bounds
    check_subset(selector.get(), 3, 5);
}