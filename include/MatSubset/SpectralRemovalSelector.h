#ifndef SPECTRAL_REMOVAL_H
#define SPECTRAL_REMOVAL_H

#include "frobenius_removal_selector.h"

namespace MatSubset {

template <typename scalar>
class SpectralRemovalSelector : public FrobeniusRemovalSelector<scalar> {
  private:
    scalar boundInternal(uint m, uint n, uint k, Norm norm) const override;

  public:
    SpectralRemovalSelector(scalar eps = 1e-6);

    std::string getAlgorithmName() const override;

    std::vector<uint> selectSubset(const Eigen::MatrixX<scalar> &x,
                                   uint k) override;
};

} // namespace MatSubset

#include "../../src/subset_selection_algorithms/spectral_removal_selector.hpp"

#endif