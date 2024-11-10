#ifndef DUAL_SET_SELECTOR_H
#define DUAL_SET_SELECTOR_H

#include "subset_selector.h"

namespace SubsetSelection {

template <typename scalar>
class DualSetSelector : public SubsetSelector<scalar> {
  public:
    DualSetSelector();

    std::vector<uint> selectSubset(const Eigen::MatrixX<scalar> &x,
                                   uint k) override;

    scalar bound(uint m, uint n, uint k, Norm norm) override;

  private:
    Eigen::ArrayX<scalar> calculateL(const Eigen::MatrixX<scalar> &V,
                                     scalar delta_l,
                                     const Eigen::MatrixX<scalar> &A, scalar l);
    Eigen::ArrayX<scalar> calculateU(scalar delta_u,
                                     const Eigen::ArrayX<scalar> &B, scalar u);
};

} // namespace SubsetSelection

#include "../../src/subset_selection_algorithms/dual_set_selector.hpp"

#endif