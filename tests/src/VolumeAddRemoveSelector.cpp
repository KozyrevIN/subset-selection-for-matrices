#include <memory>

#include <doctest/doctest.h>

#include <MatSubset/VolumeAddRemoveSelector.h>

#include "CheckBounds.h"
#include "CheckSubset.h"

TEST_CASE_TEMPLATE("volume add-remove", Scalar, float, double) {

    std::unique_ptr<MatSubset::VolumeAddRemoveSelector<Scalar>> selector =
        std::make_unique<MatSubset::VolumeAddRemoveSelector<Scalar>>(1.01);

    // Algorithm name
    CHECK(selector->getAlgorithmName() == "volume add-remove");

    // Check that returned vector is subset of column indices
    check_subset(selector.get(), 3, 5);

    // Check bounds
    check_bounds(selector.get(), 3, 5);
}

TEST_CASE_TEMPLATE("volume add-remove with greedy init", Scalar, float,
                   double) {

    std::unique_ptr<MatSubset::VolumeAddRemoveSelector<Scalar>> selector =
        std::make_unique<MatSubset::VolumeAddRemoveSelector<Scalar>>(1.01,
                                                                     true);

    // Check that returned vector is subset of column indices
    check_subset(selector.get(), 3, 5);

    // Check bounds
    check_bounds(selector.get(), 3, 5);
}

TEST_CASE_TEMPLATE("volume add-remove with greedy init and oversampling",
                   Scalar, float, double) {

    std::unique_ptr<MatSubset::VolumeAddRemoveSelector<Scalar>> selector =
        std::make_unique<MatSubset::VolumeAddRemoveSelector<Scalar>>(1.01,
                                                                     true, 1);

    // Check that returned vector is subset of column indices
    check_subset(selector.get(), 3, 4);

    // Check bounds
    check_bounds(selector.get(), 3, 4);
}
