#ifndef SPECTRAL_SELECTION_SELECTOR_H
#define SPECTRAL_SELECTION_SELECTOR_H

#include <functional>

#include "subset_selector.h"

namespace SubsetSelection {

template <typename scalar>
class SpectralSelectionSelector : public SubsetSelector<scalar> {
  private:
    scalar eps;

    scalar calculateEpsilon(const uint m, const uint n, const uint k);

    scalar calculateDelta(const uint m, const uint n, const uint k,
                          const scalar epsilon, const scalar l,
                          const uint cols_remaining_size);

    scalar binarySearch(scalar l, scalar r,
                        const std::function<scalar(scalar)> &f,
                        const scalar eps);

  public:
    SpectralSelectionSelector(scalar eps = 1e-6);

    std::vector<uint> selectSubset(const Eigen::MatrixX<scalar> &x,
                                   uint k) override;

    scalar bound(uint m, uint n, uint k, Norm norm) override;
};

} // namespace SubsetSelection

#include "../../src/subset_selection_algorithms/spectral_selection_selector.hpp"

#endif