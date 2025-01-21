#ifndef INTERLACING_FAMILIES_SELECTOR_H
#define INTERLACING_FAMILIES_SELECTOR_H

#include <functional>

#include "subset_selector.h"

namespace SubsetSelection {

template <typename scalar>
class InterlacingFamiliesSelector : public SubsetSelector<scalar> {
  private:
    scalar eps;

    long long factorial(const uint n);
    Eigen::VectorX<scalar>
    multByDeg1PolyNTimes(const Eigen::VectorX<scalar> &poly, const scalar root,
                         const uint n);
    Eigen::VectorX<scalar> polyFromRoots(const Eigen::VectorX<scalar> &roots);
    Eigen::VectorX<scalar> nThDerivative(const Eigen::VectorX<scalar> &poly,
                                         const uint n);

  public:
    InterlacingFamiliesSelector(scalar eps = 1e-6);

    std::vector<uint> selectSubset(const Eigen::MatrixX<scalar> &x,
                                   uint k) override;

    scalar bound(uint m, uint n, uint k, Norm norm) override;
};

} // namespace SubsetSelection

#include "../../src/subset_selection_algorithms/interlacing_families_selector.hpp"

#endif