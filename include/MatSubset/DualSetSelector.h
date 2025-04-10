#ifndef MAT_SUBSET_DUAL_SET_SELECTOR_H
#define MAT_SUBSET_DUAL_SET_SELECTOR_H

#include "SelectorBase.h"

namespace MatSubset {

template <typename scalar>
class DualSetSelector : public SelectorBase<scalar> {
  public:
    DualSetSelector();

    std::string getAlgorithmName() const override;

    std::vector<uint> selectSubset(const Eigen::MatrixX<scalar> &x,
                                   uint k) override;

  private:
    scalar boundInternal(uint m, uint n, uint k, Norm norm) const override;

    Eigen::ArrayX<scalar> calculateL(const Eigen::MatrixX<scalar> &V,
                                     scalar delta_l,
                                     const Eigen::MatrixX<scalar> &A,
                                     scalar l) const;

    Eigen::ArrayX<scalar>
    calculateU(scalar delta_u, const Eigen::ArrayX<scalar> &B, scalar u) const;
};

} // namespace MatSubset

#include "../../src/subset_selection_algorithms/dual_set_selector.hpp"

#endif