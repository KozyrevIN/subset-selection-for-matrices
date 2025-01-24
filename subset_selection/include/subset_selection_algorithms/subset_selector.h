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
  protected:
    virtual scalar bound(uint m, uint n, uint k, Norm norm);

  public:
    const std::string algorithmName;

    SubsetSelector(const std::string &algorithm_name);

    virtual std::vector<uint> selectSubset(const Eigen::MatrixX<scalar> &X,
                                           uint k);

    template <Norm norm> scalar bound(const Eigen::MatrixX<scalar> &X, uint k);

    // lower bound on \vert X^\dag \vert_2 / \vert X_S^\dag \vert_2
    template <Norm norm> scalar bound(uint m, uint n, uint k);
};

} // namespace SubsetSelection

#include "../../src/subset_selection_algorithms/subset_selector.hpp"

#endif