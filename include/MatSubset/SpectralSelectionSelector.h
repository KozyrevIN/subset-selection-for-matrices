#ifndef MAT_SUBSET_SPECTRAL_SELECTION_SELECTOR_H
#define MAT_SUBSET_SPECTRAL_SELECTION_SELECTOR_H

#include <functional>

#include "SelectorBase.h"

namespace MatSubset {

template <typename scalar>
class SpectralSelectionSelector : public SelectorBase<scalar> {
  private:
    scalar eps;

    scalar calculateEpsilon(uint m, uint n, uint k) const;

    scalar calculateDelta(uint m, uint n, uint k, scalar epsilon, scalar l,
                          uint cols_remaining_size) const;

    scalar binarySearch(scalar l, scalar r,
                        const std::function<scalar(scalar)> &f,
                        scalar tol) const;

    scalar boundInternal(uint m, uint n, uint k, Norm norm) const override;

  public:
    SpectralSelectionSelector(scalar eps = 1e-4);

    std::string getAlgorithmName() const override;

    std::vector<uint> selectSubset(const Eigen::MatrixX<scalar> &x,
                                   uint k) override;
};

} // namespace MatSubset

#include "../../src/subset_selection_algorithms/spectral_selection_selector.hpp"

#endif