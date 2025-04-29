#ifndef MAT_SUBSET_SUBSET_SELECTOR_H
#define MAT_SUBSET_SUBSET_SELECTOR_H

#include <string>

#include <Eigen/Core>

#include "Enums.h"

namespace MatSubset {

/*
Virtual base class for selecting k columns from m x n matrix (n >= k >= m)
*/
template <typename scalar> class SelectorBase {
  public:
    SelectorBase() {}

    virtual std::string getAlgorithmName() const { return "random columns"; }

    virtual std::vector<Eigen::Index>
    selectSubset(const Eigen::MatrixX<scalar> &X, Eigen::Index k) {
        std::vector<Eigen::Index> cols(k);
        for (Eigen::Index i = 0; i < k; ++i) {
            cols[i] = i;
        }

        return cols;
    }

    // lower bound on \vert X^\dag \vert_2^2 / \vert X_S^\dag \vert_2^2
    template <Norm norm>
    scalar bound(const Eigen::MatrixX<scalar> &X, Eigen::Index k) const {

        static_assert(norm == Norm::Frobenius || norm == Norm::L2,
                      "This norm is unsupported!");
        return boundInternal(X.rows(), X.cols(), k, norm);
    }

    template <Norm norm>
    scalar bound(Eigen::Index m, Eigen::Index n, Eigen::Index k) const {

        static_assert(norm == Norm::Frobenius || norm == Norm::L2,
                      "This norm is unsopported!");
        return boundInternal(m, n, k, norm);
    }

  private:
    virtual scalar boundInternal(Eigen::Index m, Eigen::Index n, Eigen::Index k,
                                 Norm norm) const {
        return 0;
    }
};

} // namespace MatSubset

#endif