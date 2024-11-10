#ifndef VOLUME_REMOVAL_SELECTOR
#define VOLUME_REMOVAL_SELECTOR

#include "subset_selector.h"

namespace SubsetSelection {

template <typename scalar>
class VolumeRemovalSelector : public SubsetSelector<scalar> {
  private:
    scalar eps;

  public:
    VolumeRemovalSelector(scalar eps = 1e-8);

    std::vector<uint> selectSubset(const Eigen::MatrixX<scalar> &x,
                                   uint k) override;
};

} // namespace SubsetSelection

#include "../../src/subset_selection_algorithms/volume_removal_selector.hpp"

#endif