#ifndef FROBENIUS_REMOVAL_SELECTOR_H
#define FROBENIUS_REMOVAL_SELECTOR_H

#include "subset_selector.h"

namespace SubsetSelection {

template <typename scalar>
class FrobeniusRemovalSelector : public SubsetSelector<scalar> {
  private:
    double eps;

  public:
    FrobeniusRemovalSelector(scalar eps = 1e-6);

    std::vector<uint> selectSubset(const Eigen::MatrixX<scalar> &x,
                                   uint k) override;

    scalar bound(uint m, uint n, uint k, Norm norm) override;
};

} // namespace SubsetSelection

#include "../../src/subset_selection_algorithms/frobenius_removal_selector.hpp"

#endif