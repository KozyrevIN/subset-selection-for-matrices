#ifndef MAT_SUBSET_SUBSET_SELECTOR_H
#define MAT_SUBSET_SUBSET_SELECTOR_H

#include <string>

#include <Eigen/Dense>

#include "Enums.h"

namespace MatSubset {

/*
Virtual base class for selecting k columns from m x n matrix (n >= k >= m)
*/
template <typename scalar> class SelectorBase {
  public:
    SelectorBase() {}

    virtual std::string getAlgorithmName() const { return "random columns"; }

    virtual std::vector<uint> selectSubset(const Eigen::MatrixX<scalar> &X,
                                           uint k) {
        std::vector<uint> cols(k);
        for (int i = 0; i < k; ++i) {
            cols[i] = i;
        }

        return cols;
    }

    // lower bound on \vert X^\dag \vert_2^2 / \vert X_S^\dag \vert_2^2
    template <Norm norm>
    scalar bound(const Eigen::MatrixX<scalar> &X, uint k) const {

        static_assert(norm == Norm::Frobenius || norm == Norm::L2,
                      "This norm is unsopported!");
        return boundInternal(X.rows(), X.cols(), k, norm);
    }

    template <Norm norm> scalar bound(uint m, uint n, uint k) const {

        static_assert(norm == Norm::Frobenius || norm == Norm::L2,
                      "This norm is unsopported!");
        return boundInternal(m, n, k, norm);
    }

  private:
    virtual scalar boundInternal(uint m, uint n, uint k, Norm norm) const {
        return 0;
    }
};

} // namespace MatSubset

#endif