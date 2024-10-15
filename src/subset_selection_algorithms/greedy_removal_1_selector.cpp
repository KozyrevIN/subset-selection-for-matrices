#include <vector>
#include <iostream>

#include "../../include/subset_selection_algorithms/greedy_removal_1_selector.h"

template <typename scalar>
GreedyRemoval1Selector<scalar>::GreedyRemoval1Selector(scalar eps): SubsetSelector<scalar>("greedy_removal_1"), eps(eps) {
    //do nothing
}

template <typename scalar>
std::vector<uint> GreedyRemoval1Selector<scalar>::selectSubset(const Eigen::MatrixX<scalar>& X, uint k) {
    uint m = X.rows();
    uint n = X.cols();

    std::vector<uint> cols(n);
    for (uint j = 0; j < n; ++j) {
        cols[j] = j;
    }

    if (k < n) {
        Eigen::JacobiSVD<Eigen::MatrixX<scalar>> svd(X, Eigen::ComputeThinV);
        Eigen::MatrixX<scalar> V = svd.matrixV().transpose(); 

        Eigen::MatrixX<scalar> VVT_invV = (V * V.transpose()).inverse() * V;
        Eigen::ArrayX<scalar> d = 1 - (V.transpose() * VVT_invV).diagonal().array();

        for (uint cols_remaining = n; cols_remaining >= k; --cols_remaining) {
            uint j_max;
            scalar d_max = d.head(cols_remaining).maxCoeff(&j_max);

            std::swap(cols[j_max], cols[cols_remaining - 1]);
            std::swap(d(j_max), d(cols_remaining - 1));
            V.col(j_max).swap(V.col(cols_remaining - 1));
            VVT_invV.col(j_max).swap(VVT_invV.col(cols_remaining - 1));

            d.head(cols_remaining - 1) -= (V.col(cols_remaining - 1).transpose() * 
                                      VVT_invV.leftCols(cols_remaining - 1)).array().square() / d_max;
            VVT_invV.leftCols(cols_remaining - 1) += VVT_invV.col(cols_remaining - 1) *
                                                 (VVT_invV.col(cols_remaining - 1).transpose() *
                                                 V.leftCols(cols_remaining - 1)) / d_max;
        }
    }

    cols.resize(k);
    return cols;
}

template class GreedyRemoval1Selector<float>;
template class GreedyRemoval1Selector<double>;