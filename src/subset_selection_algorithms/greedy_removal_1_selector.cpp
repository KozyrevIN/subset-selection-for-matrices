#include <vector>
#include <iostream>

#include "../../include/subset_selection_algorithms/greedy_removal_1_selector.h"

template <typename scalar>
GreedyRemoval1Selector<scalar>::GreedyRemoval1Selector(scalar eps): SubsetSelector<scalar>("greedy_removal_1"), eps(eps) {
    //do nothing
}

template<typename scalar>
inline constexpr auto remove_col(const Eigen::MatrixX<scalar>& matrix, const int& col_num)
{
    return (Eigen::MatrixX<scalar>(matrix.rows(), matrix.cols() - 1)
        << static_cast<Eigen::MatrixX<scalar>>(matrix.leftCols(col_num - 1)),
        static_cast<Eigen::MatrixX<scalar>>(matrix.rightCols(matrix.cols() - col_num))).finished();
}

template<typename scalar>
inline constexpr auto remove_col(const Eigen::VectorX<scalar>& vector, const int& row_num)
{
    return (Eigen::VectorX<scalar>(vector.rows() - 1)
        << static_cast<Eigen::VectorX<scalar>>(vector.head(row_num - 1)),
        static_cast<Eigen::MatrixX<scalar>>(vector.tail(vector.rows() - row_num))).finished();
}

/*
template <typename scalar>
std::vector<uint> GreedyRemoval1Selector<scalar>::selectSubset(const Eigen::MatrixX<scalar>& X, uint k) {
    std::vector<uint> cols(X.cols());
    for (uint j = 0; j < X.cols(); ++j) {
        cols[j] = j;
    }

    if (k < cols.size()) {
        Eigen::MatrixX<scalar> Y = X;
        Eigen::MatrixX<scalar> XXT_inv = (X * X.transpose()).inverse();
        Eigen::MatrixX<scalar> XXT_invx;
        Eigen::VectorX<scalar> d(cols.size());
        scalar d_max = 0.0; uint j_max;
        scalar d_max_prev;

        for (uint j = 0; j < cols.size(); ++j) {
            d(j) = 1.0 - (Y.col(cols[j]).transpose() * XXT_inv * Y.col(cols[j])).value();
            if (d(j) > eps and d(j) > d_max) {
                d_max = d(j);
                j_max = j;
            }
        }

        XXT_invx = XXT_inv * Y.col(j_max);
        XXT_inv += (XXT_inv * Y.col(j_max)) * (Y.col(j_max).transpose() * XXT_inv) / d_max;
        cols.erase(cols.begin() + j_max);
        d_max_prev = d_max;
        d_max = 0.0;
        remove_col(Y, j_max);
        remove_col(d, j_max);

        while (cols.size() > k) {
            d -= (Y.transpose() * XXT_invx).cwiseAbs2() / d_max_prev;
            d_max = d.maxCoeff(&j_max);

            XXT_invx = XXT_inv * Y.col(j_max);
            XXT_inv += (XXT_inv * Y.col(j_max)) * (Y.col(j_max).transpose() * XXT_inv) / d_max;
            d_max_prev = d_max;
            d_max = 0.0;
            cols.erase(cols.begin() + j_max);
            remove_col(Y, j_max);
            remove_col(d, j_max);
        }
    }

    return cols;
}*/

template <typename scalar>
std::vector<uint> GreedyRemoval1Selector<scalar>::selectSubset(const Eigen::MatrixX<scalar>& X, uint k) {
    std::vector<uint> cols(X.cols());
    for (uint j = 0; j < X.cols(); ++j) {
        cols[j] = j;
    }

    if (k < cols.size()) {
        Eigen::MatrixX<scalar> XXT_inv = (X * X.transpose()).inverse();
        Eigen::MatrixX<scalar> XXT_invx;
        Eigen::VectorX<scalar> d(cols.size());
        scalar d_max = 0.0; uint j_max;
        scalar d_max_prev;

        //#pragma omp parallel for
        for (uint j = 0; j < cols.size(); ++j) {
            d(j) = 1.0 - (X.col(cols[j]).transpose() * XXT_inv * X.col(cols[j])).value();
            if (d(j) > eps and d(j) > d_max) {
                d_max = d(j);
                j_max = j;
            }
        }

        XXT_invx = XXT_inv * X.col(cols[j_max]) / d_max;
        XXT_inv += (XXT_inv * X.col(cols[j_max])) * (X.col(cols[j_max]).transpose() * XXT_inv) / d_max;
        d_max_prev = d_max;
        d_max = 0.0;
        cols.erase(cols.begin() + j_max);

        while (cols.size() > k) {
            //#pragma omp parallel for
            for (uint j = 0; j < cols.size(); ++j) {
                d(cols[j]) -= (X.col(cols[j]).transpose() * XXT_invx).cwiseAbs2().value() / d_max_prev;
                if (d(cols[j]) > d_max and d(cols[j]) > eps) {
                    d_max = d(cols[j]);
                    j_max = j;
                }            
            }

            XXT_invx = XXT_inv * X.col(cols[j_max]);
            XXT_inv += XXT_inv * X.col(cols[j_max]) * X.col(cols[j_max]).transpose() * XXT_inv / d_max;
            d_max_prev = d_max;
            d_max = 0.0;
            cols.erase(cols.begin() + j_max);
        }
    }

    return cols;
}

template class GreedyRemoval1Selector<float>;
template class GreedyRemoval1Selector<double>;