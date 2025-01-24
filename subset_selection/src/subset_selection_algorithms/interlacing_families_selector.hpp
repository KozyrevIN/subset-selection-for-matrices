#include <eigen3/Eigen/Eigenvalues>
#include <eigen3/Eigen/SVD>
#include <eigen3/unsupported/Eigen/Polynomials>
#include <functional>
#include <iostream>
#include <iomanip>
#include <vector>

namespace SubsetSelection {

template <typename scalar>
InterlacingFamiliesSelector<scalar>::InterlacingFamiliesSelector(scalar eps)
    : SubsetSelector<scalar>("interlacing_families"), eps(eps) {
    // do nothing
}

template <typename scalar>
Eigen::VectorX<scalar> InterlacingFamiliesSelector<scalar>::polyFromRoots(
    const Eigen::VectorX<scalar> &roots) {

    uint l = roots.size();
    Eigen::VectorX<scalar> poly = Eigen::VectorX<scalar>::Zero(l + 1);
    poly(l) = 1;

    for (scalar root : roots) {
        poly.head(l) -= root * poly.tail(l);
    }

    return poly;
}

template <typename scalar>
void InterlacingFamiliesSelector<scalar>::fYFromPY(Eigen::VectorX<scalar> &p_y,
                                                   const uint m, const uint n,
                                                   const uint k, const uint i) {
    if (k <= n - m) {
        scalar coeff = 1;
        for (uint j = 1; j < p_y.size(); ++j) {
            coeff *= j + n - m - i;
            coeff /= j + n - m - k;
            p_y(j) *= coeff;
        }
    } else {
        p_y = p_y.tail(n - k + 1);
        scalar coeff = 1;
        for (uint j = 1; j < p_y.size(); ++j) {
            coeff *= j + k - i;
            coeff /= j;
        }
    }
}

template <typename scalar>
std::vector<uint> InterlacingFamiliesSelector<scalar>::selectSubset(
    const Eigen::MatrixX<scalar> &X, uint k) {
    uint m = X.rows();
    uint n = X.cols();

    Eigen::BDCSVD svd(X, Eigen::ComputeThinV);
    Eigen::MatrixX<scalar> V = svd.matrixV().transpose();
    Eigen::MatrixX<scalar> Y = Eigen::MatrixX<scalar>::Zero(m, m);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<scalar>> decomposition(m);
    Eigen::PolynomialSolver<scalar, Eigen::Dynamic> poly_solver;

    std::vector<uint> cols_remaining(n);
    for (uint j = 0; j < X.cols(); ++j) {
        cols_remaining[j] = j;
    }

    std::vector<uint> cols_selected;
    cols_selected.reserve(k);

    for (uint i = 1; i <= k; ++i) {
        Eigen::VectorX<scalar> lambdas(cols_remaining.size());

        for (uint j = 0; j < cols_remaining.size(); ++j) {
            decomposition.compute(Y + V.col(j) * V.col(j).transpose(), Eigen::EigenvaluesOnly);
            Eigen::VectorX<scalar> p_roots = decomposition.eigenvalues();
            // y = x - 1
            Eigen::VectorX<scalar> p_roots_y = p_roots.array() - 1;
            Eigen::VectorX<scalar> p_y = polyFromRoots(p_roots_y);
            fYFromPY(p_y, m, n, k, i);
            
            poly_solver.compute(p_y);

            bool has_root;
            lambdas(j) = poly_solver.smallestRealRoot(has_root, 0.01);
        }

        uint j_max;
        lambdas.maxCoeff(&j_max);
        Y += V.col(j_max) * V.col(j_max).transpose();

        decomposition.compute(Y);

        cols_selected.push_back(cols_remaining[j_max]);
        cols_remaining[j_max] = cols_remaining.back();
        cols_remaining.pop_back();
        V.col(j_max) = V.col(V.cols() - 1);
        V.conservativeResize(Eigen::NoChange, V.cols() - 1);
    }

    return cols_selected;
}

template <typename scalar>
scalar InterlacingFamiliesSelector<scalar>::bound(uint m, uint n, uint k,
                                                  Norm norm) {
    return (std::sqrt((k + 1) * (n - m)) - std::sqrt(m * (n - k - 1))) / n
}

} // namespace SubsetSelection