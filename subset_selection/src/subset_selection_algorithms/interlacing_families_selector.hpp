#include <eigen3/Eigen/Eigenvalues>
#include <eigen3/Eigen/SVD>
#include <eigen3/unsupported/Eigen/Polynomials>
#include <functional>
#include <iostream>
#include <vector>

namespace SubsetSelection {

template <typename scalar>
InterlacingFamiliesSelector<scalar>::InterlacingFamiliesSelector(scalar eps)
    : SubsetSelector<scalar>("interlacing_families"), eps(eps) {
    // do nothing
}

template <typename scalar>
long long InterlacingFamiliesSelector<scalar>::factorial(const uint n) {
    long long f = 1;
    for (int i = 1; i <= n; ++i)
        f *= i;
    return f;
}

template <typename scalar>
Eigen::VectorX<scalar>
InterlacingFamiliesSelector<scalar>::multByDeg1PolyNTimes(
    const Eigen::VectorX<scalar> &poly, const scalar root, const uint n) {
    uint l = poly.size();
    Eigen::VectorX<scalar> new_poly = Eigen::VectorX<scalar>::Zero(l + n);
    new_poly.tail(l) = poly;

    for (int i = 0; i < n; ++i) {
        new_poly.head(l + n - 1) -= root * new_poly.tail(n + l - 1);
    }

    return new_poly;
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
Eigen::VectorX<scalar> InterlacingFamiliesSelector<scalar>::nThDerivative(
    const Eigen::VectorX<scalar> &poly, const uint n) {
    uint l = poly.size();
    Eigen::VectorX<scalar> new_poly = poly.tail(l - n);

    long long factor = factorial(n);
    for (uint i = 0; i < l - n; i++) {
        new_poly(i) *= factor;
        factor *= i + n + 1;
        factor /= i + 1;
    }

    return new_poly;
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

    for(uint i = 1; i <= k; ++i) {
        Eigen::VectorX<scalar> lambdas(cols_remaining.size());

        for (uint j = 0; j < cols_remaining.size(); ++j) {
            decomposition.compute(Y + V.col(j) * V.col(j).transpose());
            Eigen::VectorX<scalar> eigenvalues = decomposition.eigenvalues();
            Eigen::VectorX<scalar> poly = polyFromRoots(eigenvalues);
            poly = multByDeg1PolyNTimes(poly, 1, n - m - i);
            poly = nThDerivative(poly, k - i);
            poly_solver.compute(poly);

            bool has_root;
            lambdas(j) = poly_solver.smallestRealRoot(has_root, eps);
        }

        uint j_max;
        lambdas.maxCoeff(&j_max);
        Y += V.col(j_max) * V.col(j_max).transpose();

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
    return (std::pow(k, 0.5) - std::pow(m - 1, 0.5)) / std::pow(n, 0.5);
}

} // namespace SubsetSelection