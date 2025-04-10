#ifndef MAT_SUBSET_INTERLACING_FAMILIES_SELECTOR_H
#define MAT_SUBSET_INTERLACING_FAMILIES_SELECTOR_H

#include <functional>

#include "SelectorBase.h"

namespace MatSubset {

template <typename scalar>
class InterlacingFamiliesSelector : public SelectorBase<scalar> {
  private:
    scalar eps;

    Eigen::VectorX<scalar>
    polyFromRoots(const Eigen::VectorX<scalar> &roots) const;

    Eigen::ArrayX<scalar> PtoFArray(uint m, uint n, uint k, uint i) const;

    Eigen::MatrixX<scalar> YtoZMatrix(uint m, scalar shift) const;

    scalar boundInternal(uint m, uint n, uint k, Norm norm) const override;

  public:
    InterlacingFamiliesSelector(scalar eps = 1e-2);

    std::string getAlgorithmName() const override;

    std::vector<uint> selectSubset(const Eigen::MatrixX<scalar> &x,
                                   uint k) override;
};

} // namespace MatSubset

#include "../../src/subset_selection_algorithms/interlacing_families_selector.hpp"

#endif