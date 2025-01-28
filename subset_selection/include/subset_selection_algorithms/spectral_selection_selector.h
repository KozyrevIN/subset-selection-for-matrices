#ifndef SPECTRAL_SELECTION_SELECTOR_H
#define SPECTRAL_SELECTION_SELECTOR_H

#include <functional>

#include "subset_selector.h"

namespace SubsetSelection {

template <typename scalar>
class SpectralSelectionSelector : public SubsetSelector<scalar> {
  private:
    scalar eps;

    scalar calculateEpsilon(uint m, uint n, uint k) const;

    scalar calculateDelta(uint m, uint n, uint k, scalar epsilon,
                                 scalar l, uint cols_remaining_size) const;

    scalar binarySearch(scalar l, scalar r,
                        const std::function<scalar(scalar)> &f) const;

    scalar boundInternal(uint m, uint n, uint k, Norm norm) const override;

  public:
    SpectralSelectionSelector(scalar eps = 1e-6);

    std::string getAlgorithmName() const override;

    std::vector<uint> selectSubset(const Eigen::MatrixX<scalar> &x,
                                   uint k) override;
};

} // namespace SubsetSelection

#include "../../src/subset_selection_algorithms/spectral_selection_selector.hpp"

#endif