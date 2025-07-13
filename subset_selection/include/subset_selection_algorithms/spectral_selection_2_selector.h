#ifndef SPECTRAL_SELECTION_2_SELECTOR_H
#define SPECTRAL_SELECTION_2_SELECTOR_H

#include <functional>
#include <tuple>

#include "subset_selector.h"

namespace SubsetSelection {

template <typename scalar>
class SpectralSelection2Selector : public SubsetSelector<scalar> {
  private:
    scalar eps;

    scalar computeEpsilon(scalar l,
                          const Eigen::ArrayX<scalar> &eigenvalues) const;

    scalar computeDelta(uint m, uint i, uint n, scalar l, scalar epsilon) const;

    scalar computeBound(uint m, uint i, uint k, uint n, scalar l,
                        const Eigen::ArrayX<scalar> &eigenvalues) const;

    scalar optimizeBound(uint m, uint i, uint k, uint n,
                         const Eigen::ArrayX<scalar> &eigenvalues) const;

    scalar binarySearch(scalar l, scalar r,
                        const std::function<scalar(scalar)> &f,
                        scalar tol) const;

    scalar boundInternal(uint m, uint n, uint k, Norm norm) const override;

  public:
    SpectralSelection2Selector(scalar eps = 1e-8);

    std::string getAlgorithmName() const override;

    std::vector<uint> selectSubset(const Eigen::MatrixX<scalar> &X,
                                   uint k) override;
};

} // namespace SubsetSelection

#include "../../src/subset_selection_algorithms/spectral_selection_2_selector.hpp"

#endif