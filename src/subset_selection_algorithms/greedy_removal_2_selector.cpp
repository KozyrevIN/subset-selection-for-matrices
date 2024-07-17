#include <vector>
#include <unordered_set>
#include <float.h>

#include "../../include/subset_selection_algorithms/greedy_removal_2_selector.h"

template <typename scalar>
GreedyRemoval2Selector<scalar>::GreedyRemoval2Selector(double eps): SubsetSelector<scalar>("greedy_removal_2"), eps(eps) {
    //do nothing
}

template <typename scalar>
std::vector<uint> GreedyRemoval2Selector<scalar>::selectSubset(const Eigen::MatrixX<scalar>& X, uint k) {
    std::unordered_set<uint> cols;
    for (uint i = 0; i < X.cols(); ++i) {
        cols.insert(i);
    }

    Eigen::MatrixX<scalar> XXT_inv = (X * X.adjoint()).inverse();
    Eigen::MatrixX<scalar> XXT_inv2 = XXT_inv * XXT_inv;

    while (cols.size() > k) {
        double min_expr = DBL_MAX;
        auto min_it = cols.begin();
        for (auto it = cols.begin(); it != cols.end(); ++it) {
            double numerator = (X.col(*it).adjoint() * XXT_inv2 * X.col(*it)).value();
            double denominator = 1 - (X.col(*it).adjoint() * XXT_inv * X.col(*it)).value();
            if (numerator > eps and  numerator / denominator < min_expr) {
                min_expr = numerator / denominator;
                min_it = it;
            }
        }

        //recalculating matrixes
        auto x = X.col(*min_it);
        double denominator = 1 - (x.adjoint() * XXT_inv * x).value();
        auto XXT_invx = XXT_inv * x;
        auto XXT_inv2x = XXT_inv2 * x;
        auto XXT_inv2xxTXXT_inv_div = XXT_inv2x * XXT_invx.adjoint() / denominator;
        auto XXT_invxxTXXT_inv_div = XXT_invx * XXT_invx.adjoint() / denominator;
        XXT_inv += XXT_invxxTXXT_inv_div;
        XXT_inv2 += XXT_inv2xxTXXT_inv_div + XXT_inv2xxTXXT_inv_div.adjoint() + 
                    XXT_invxxTXXT_inv_div * (x.adjoint() * XXT_inv2x).value() / denominator;
        cols.erase(min_it);
    }

    std::vector<uint> vec;
    std::copy(cols.begin(), cols.end(), std::back_inserter(vec));
    std::sort(vec.begin(), vec.end());
    return vec;
}

template class GreedyRemoval2Selector<float>;
template class GreedyRemoval2Selector<double>;
//template class GreedyRemoval2Selector<std::complex<float>>;
//template class GreedyRemoval2Selector<std::complex<double>>;