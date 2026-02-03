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
}
