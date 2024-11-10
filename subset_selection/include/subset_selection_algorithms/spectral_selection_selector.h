#ifndef SPECTRAL_SELECTION_SELECTOR_H
#define SPECTRAL_SELECTION_SELECTOR_H

#include "subset_selector.h"

namespace SubsetSelection {

template <typename scalar>
class SpectralSelectionSelector : public SubsetSelector<scalar> {
  public:
    SpectralSelectionSelector();

    std::vector<uint> selectSubset(const Eigen::MatrixX<scalar> &x,
                                   uint k) override;

    scalar bound(uint m, uint n, uint k, Norm norm) override;
};

} // namespace SubsetSelection

#include "../../src/subset_selection_algorithms/spectral_selection_selector.hpp"

#endif