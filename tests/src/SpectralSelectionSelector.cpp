#include <memory>

#include <doctest/doctest.h>

#include <MatSubset/SpectralSelectionSelector.h>

#include "CheckBounds.h"
#include "CheckSubset.h"

TEST_CASE_TEMPLATE("spectral selection", Scalar, float, double) {

    std::unique_ptr<MatSubset::SpectralSelectionSelector<Scalar>> selector =
        std::make_unique<MatSubset::SpectralSelectionSelector<Scalar>>();

    // Algorithm name
    CHECK(selector->getAlgorithmName() == "spectral selection");

    // Check that returned vector is subset of column indices
    check_subset(selector.get(), 3, 5);

    // Check bounds
    check_subset(selector.get(), 3, 5);
}