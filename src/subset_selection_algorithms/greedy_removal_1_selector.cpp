#include <vector>
#include <unordered_set>
#include <complex>
#include <iostream>
#include <float.h>

#include "../../include/subset_selection_algorithms/greedy_removal_1_selector.h"

template <typename scalar>
GreedyRemoval1Selector<scalar>::GreedyRemoval1Selector(scalar eps): SubsetSelector<scalar>("greedy_removal_1"), eps(eps) {
    //do nothing
}

template <typename scalar>
std::vector<uint> GreedyRemoval1Selector<scalar>::selectSubset(const Eigen::MatrixX<scalar>& X, uint k) {
    std::vector<uint> cols(X.cols());
    for (uint j = 0; j < X.cols(); ++j) {
        cols[j] = j;
    }

    if (k < cols.size()) {
        Eigen::VectorX<scalar> x;
        Eigen::MatrixX<scalar> XXT_inv = (X * X.transpose()).inverse();
        Eigen::MatrixX<scalar> XXT_invx_d_max_m1;
        Eigen::VectorX<scalar> d(cols.size());
        scalar d_max = 0.0; uint j_max;

        //#pragma omp parallel for
        for (uint j = 0; j < cols.size(); ++j) {
            d(j) = 1.0 - (X.col(cols[j]).transpose() * XXT_inv * X.col(cols[j])).value();
            if (d(j) > eps and d(j) > d_max) {
                d_max = d(j);
                j_max = j;
            }
        }

        XXT_invx_d_max_m1 = XXT_inv * X.col(cols[j_max]) / d_max;
        XXT_inv += (XXT_inv * X.col(cols[j_max])) * (X.col(cols[j_max]).transpose() * XXT_inv) / d_max;
        d_max = 0.0;
        cols.erase(cols.begin() + j_max);

        while (cols.size() > k) {
            //#pragma omp parallel for
            for (uint j = 0; j < cols.size(); ++j) {
                d(cols[j]) -= (X.col(cols[j]).transpose() * XXT_invx_d_max_m1).value();
                if (d(cols[j]) > d_max and d(cols[j]) > eps) {
                    d_max = d(cols[j]);
                    j_max = j;
                }            
            }

            XXT_invx_d_max_m1 = XXT_inv * X.col(cols[j_max]) / d_max;
            XXT_inv += XXT_inv * X.col(cols[j_max]) * X.col(cols[j_max]).transpose() * XXT_inv / d_max;
            d_max = 0.0;
            cols.erase(cols.begin() + j_max);
        }
    }

    return cols;
}

template class GreedyRemoval1Selector<float>;
template class GreedyRemoval1Selector<double>;