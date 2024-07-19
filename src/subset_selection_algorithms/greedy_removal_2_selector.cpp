#include <vector>

#include "../../include/subset_selection_algorithms/greedy_removal_2_selector.h"

template <typename scalar>
GreedyRemoval2Selector<scalar>::GreedyRemoval2Selector(scalar eps): SubsetSelector<scalar>("greedy_removal_2"), eps(eps) {
    //do nothing
}

template <typename scalar>
std::vector<uint> GreedyRemoval2Selector<scalar>::selectSubset(const Eigen::MatrixX<scalar>& X, uint k) {
    std::vector<uint> cols(X.cols());
    for (uint j = 0; j < X.cols(); ++j) {
        cols[j] = j;
    }

    if (k < cols.size()) {
        Eigen::VectorX<scalar> x;
        Eigen::MatrixX<scalar> XXT_inv = (X * X.adjoint()).inverse();
        Eigen::MatrixX<scalar> XXT_inv2 = XXT_inv * XXT_inv;
        scalar xTXXT_inv2x;
        Eigen::MatrixX<scalar> XXT_invx, XXT_inv2x;
        Eigen::VectorX<scalar> l(cols.size()), d(cols.size());
        scalar l_min = 1.0, d_min = 0; scalar d_min_prev;
        uint j_min;

        //#pragma omp parallel for
        for (uint j = 0; j < cols.size(); ++j) {
            l(j) = (X.col(j).adjoint() * XXT_inv2 * X.col(j)).value();
            d(j) = 1.0 - (X.col(cols[j]).adjoint() * XXT_inv * X.col(j)).value();
            if (d(j) > eps and l(j) * d_min < l_min * d(j)) {
                l_min = l(j);
                d_min = d(j);
                j_min = j;
            }
        }

        d_min_prev = d_min;
        d_min = 0.0;
        x = X.col(cols[j_min]);
        XXT_invx = XXT_inv * x;
        XXT_inv2x = XXT_inv2 * x;
        xTXXT_inv2x = (x.adjoint() * XXT_inv2x).value();
        XXT_inv += XXT_inv * x * x.adjoint() * XXT_inv / d_min_prev;
        XXT_inv2 += 2 * (XXT_inv2x * XXT_invx.adjoint()).real() / d_min_prev +
                    XXT_invx * XXT_invx.adjoint() * (x.adjoint() * XXT_inv2x).value() / (d_min_prev * d_min_prev);
        cols.erase(cols.begin() + j_min);

        while (cols.size() > k) {
            //#pragma omp parallel for
            for (uint j = 0; j < cols.size(); ++j) {
                l(cols[j]) += 2 * std::real(((X.col(cols[j]).adjoint() * XXT_inv2x) * (XXT_invx.adjoint() * X.col(cols[j]))).value()) / d_min_prev +
                              (X.col(cols[j]).adjoint() * XXT_invx).cwiseAbs2().value() * xTXXT_inv2x / (d_min_prev * d_min_prev);
                d(cols[j]) -= (X.col(cols[j]).adjoint() * XXT_invx).cwiseAbs2().value() / d_min_prev;
                if (d(cols[j]) > eps and l(cols[j]) * d_min < l_min * d(cols[j])) {
                    l_min = l(cols[j]);
                    d_min = d(cols[j]);
                    j_min = j;
                }           
            }

            d_min_prev = d_min;
            d_min = 0.0;
            x = X.col(cols[j_min]);
            XXT_invx = XXT_inv * x;
            XXT_inv2x = XXT_inv2 * x;
            xTXXT_inv2x = (x.adjoint() * XXT_inv2x).value();
            XXT_inv += XXT_inv * x * x.adjoint() * XXT_inv / d_min_prev;
            XXT_inv2 += 2 * (XXT_inv2x * XXT_invx.adjoint()).real() / d_min_prev +
                        XXT_invx * XXT_invx.adjoint() * xTXXT_inv2x / (d_min_prev * d_min_prev);
            cols.erase(cols.begin() + j_min);
        }
    }

    return cols;
}

/*
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
}*/

template class GreedyRemoval2Selector<float>;
template class GreedyRemoval2Selector<double>;
//template class GreedyRemoval2Selector<std::complex<float>>;
//template class GreedyRemoval2Selector<std::complex<double>>;