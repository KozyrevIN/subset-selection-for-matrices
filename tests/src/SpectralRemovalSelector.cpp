#include <memory>

#include <doctest/doctest.h>

#include <MatSubset/SpectralRemovalSelector.h>

#include "CheckBounds.h"
#include "CheckSubset.h"

TEST_CASE_TEMPLATE("spectral removal", Scalar, float, double) {

    std::unique_ptr<MatSubset::SpectralRemovalSelector<Scalar>> selector =
        std::make_unique<MatSubset::SpectralRemovalSelector<Scalar>>();

    // Algorithm name
    CHECK(selector->getAlgorithmName() == "spectral removal");

    // Check that returned vector is subset of column indices
    check_subset(selector.get(), 3, 5);

    // Check bounds
    check_subset(selector.get(), 3, 5);
}