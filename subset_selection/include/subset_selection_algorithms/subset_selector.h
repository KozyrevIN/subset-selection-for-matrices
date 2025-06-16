#ifndef SUBSET_SELECTOR_H
#define SUBSET_SELECTOR_H

#include <eigen3/Eigen/Dense>
#include <string>

#include "../enums.h"

namespace SubsetSelection {

/*
Virtual base class for selecting k columns from m x n matrix (n >= k >= m)
*/
template <typename scalar> class SubsetSelector {
  private:
    virtual scalar boundInternal(uint m, uint n, uint k, Norm norm) const;

  public:
    SubsetSelector();

    virtual ~SubsetSelector() = default;

    virtual std::string getAlgorithmName() const;

    virtual std::vector<uint> selectSubset(const Eigen::MatrixX<scalar> &X,
                                           uint k);

    // lower bound on \vert X^\dag \vert_2^2 / \vert X_S^\dag \vert_2^2
    template <Norm norm>
    scalar bound(const Eigen::MatrixX<scalar> &X, uint k) const;

    template <Norm norm> scalar bound(uint m, uint n, uint k) const;
};

} // namespace SubsetSelection

#include "../../src/subset_selection_algorithms/subset_selector.hpp"

#endif