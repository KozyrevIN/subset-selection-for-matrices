#ifndef FROBENIUS_REMOVAL_SELECTOR_H
#define FROBENIUS_REMOVAL_SELECTOR_H

#include "subset_selector.h"

namespace SubsetSelection {

template <typename scalar>
class FrobeniusRemovalSelector : public SubsetSelector<scalar> {
  private:
    scalar eps;

    void removeByIdx(std::vector<uint> &cols, Eigen::ArrayX<scalar> &l,
                     Eigen::ArrayX<scalar> &d, Eigen::MatrixX<scalar> &V,
                     Eigen::MatrixX<scalar> &V_dag, uint j) const;

    static scalar boundInternal(uint m, uint n, uint k, Norm norm) override;

  public:
    FrobeniusRemovalSelector(scalar eps = 1e-6);

    std::string getAlgorithmName() const override;

    std::vector<uint> selectSubset(const Eigen::MatrixX<scalar> &x,
                                   uint k) override;
};

} // namespace SubsetSelection

#include "../../src/subset_selection_algorithms/frobenius_removal_selector.hpp"

#endif