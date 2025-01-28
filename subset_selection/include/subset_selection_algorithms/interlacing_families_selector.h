#ifndef INTERLACING_FAMILIES_SELECTOR_H
#define INTERLACING_FAMILIES_SELECTOR_H

#include <functional>

#include "subset_selector.h"

namespace SubsetSelection {

template <typename scalar>
class InterlacingFamiliesSelector : public SubsetSelector<scalar> {
  private:
    scalar eps;

    Eigen::VectorX<scalar> polyFromRoots(const Eigen::VectorX<scalar> &roots) const;

    void fYFromPY(Eigen::VectorX<scalar> &p_y, const uint m, const uint n,
                  const uint k, const uint i) const;

    static scalar boundInternal(uint m, uint n, uint k, Norm norm) override;

  public:
    InterlacingFamiliesSelector(scalar eps = 1e-2);

    std::string getAlgorithmName() const override;

    std::vector<uint> selectSubset(const Eigen::MatrixX<scalar> &x,
                                   uint k) override;
};

} // namespace SubsetSelection

#include "../../src/subset_selection_algorithms/interlacing_families_selector.hpp"

#endif