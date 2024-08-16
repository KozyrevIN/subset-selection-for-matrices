#include "../../include/subset_selection_algorithms/brute_force_selector.h"

template <typename scalar> 
BruteForceSelector<scalar>::BruteForceSelector(): SubsetSelector<scalar>("brute_force") {
    //do nothing
}

template <typename scalar>
std::vector<uint> GreedySelection1Selector<scalar>::selectSubset(const Eigen::MatrixX<scalar>& X, uint k) {
    std::vector<uint> cols;
    cols.reserve(k);
    std::vector<uint> remaining_cols(X.cols());
    for (uint j = 0; j < X.cols(); ++j) {
        remaining_cols[j] = j;
    }

    uint j_max;
    scalar val_max = 0;
    for(uint j = 0; j < X.cols(); ++j) {
        scalar val = X.col(j).squaredNorm();
        if (val > val_max) {
            val = val_max;
            j_max = j;
        }
    }
    cols.push_back(j_max);
    remaining_cols.erase(remaining_cols.begin() + j_max);

    while (cols.size() < k) {
        Eigen::JacobiSVD<Eigen::MatrixX<scalar>> svd(X(Eigen::all, cols), Eigen::ComputeThinU);
        std::cout << cols.size() << "\n";

        for (uint j = 0; j < remaining_cols.size(); ++j) {
            scalar val = X.col(remaining_cols[j]).squaredNorm() - (X.col(remaining_cols[j]).transpose() * svd.matrixU()).squaredNorm();
            if (val > val_max) {
                val_max = val;
                j_max = j;
            }
        }

        cols.push_back(remaining_cols[j_max]);
        remaining_cols.erase(remaining_cols.begin() + j_max);
    }
    
    return cols;
}

template class GreedySelection1Selector<float>;
template class GreedySelection1Selector<double>;