#include <cassert>
#include <iostream>

namespace SubsetSelection {

// MatrixGenerator class

template <typename scalar>
MatrixGenerator<scalar>::MatrixGenerator(uint m, uint n) : m(m), n(n) {
    std::random_device rd;
    gen = std::mt19937(rd());
}

template <typename scalar>
MatrixGenerator<scalar>::MatrixGenerator(uint m, uint n, int seed)
    : m(m), n(n) {
    gen = std::mt19937(seed);
}

template <typename scalar>
std::pair<uint, uint> MatrixGenerator<scalar>::getMatrixSize() {
    return std::make_pair(m, n);
}

template <typename scalar>
Eigen::MatrixX<scalar> MatrixGenerator<scalar>::generateMatrix() {
    return Eigen::MatrixX<scalar>(m, n);
}

// OrthonormalEntriesMatrixGenerator class

template <typename scalar>
OrthonormalEntriesMatrixGenerator<scalar>::OrthonormalEntriesMatrixGenerator(
    uint m, uint n)
    : MatrixGenerator<scalar>(m, n) {}

template <typename scalar>
OrthonormalEntriesMatrixGenerator<scalar>::OrthonormalEntriesMatrixGenerator(
    uint m, uint n, int seed)
    : MatrixGenerator<scalar>(m, n, seed) {}

template <typename scalar>
Eigen::MatrixX<scalar>
OrthonormalEntriesMatrixGenerator<scalar>::generateMatrix() {

    uint m = MatrixGenerator<scalar>::m;
    uint n = MatrixGenerator<scalar>::n;

    bool orthonormal_rows = false;
    if (m < n) {
        orthonormal_rows = true;
        std::swap(m, n);
    }

    std::normal_distribution<scalar> dis(0.0, 1.0);
    Eigen::MatrixX<scalar> tmp(m, n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            tmp(i, j) = dis(MatrixGenerator<scalar>::gen);
        }
    }

    Eigen::HouseholderQR<Eigen::MatrixX<scalar>> qr(tmp);
    Eigen::MatrixX<scalar> Qfull = qr.householderQ();
    Eigen::MatrixX<scalar> Q = Qfull.leftCols(n);
    Eigen::VectorX<scalar> Rdiag = qr.matrixQR().diagonal();

    for (int i = 0; i < n; ++i) {
        if (std::abs(Rdiag(i) != 0)) {
            Q.row(i) *= Rdiag(i) / std::abs(Rdiag(i));
        }
    }

    if (orthonormal_rows) {
        return Q.transpose();
    } else {
        return Q;
    }
}

// SigmaMatrixGenerator class

template <typename scalar>
SigmaMatrixGenerator<scalar>::SigmaMatrixGenerator(
    uint m, uint n, const Eigen::VectorX<scalar> &sigma)
    : MatrixGenerator<scalar>(m, n), sigma(sigma) {

    assert(sigma.size() <= std::min(m, n) &&
           "Invalid number of singular values, must be less or equal to the "
           "smallest side of a matrix");
}

template <typename scalar>
SigmaMatrixGenerator<scalar>::SigmaMatrixGenerator(
    uint m, uint n, int seed, const Eigen::VectorX<scalar> &sigma)
    : MatrixGenerator<scalar>(m, n, seed), sigma(sigma) {

    assert(sigma.size() <= std::min(m, n) &&
           "Invalid number of singular values, must be less or equal to the "
           "smallest side of a matrix");
}

template <typename scalar>
Eigen::MatrixX<scalar> SigmaMatrixGenerator<scalar>::generateMatrix() {
    uint m = MatrixGenerator<scalar>::m;
    uint n = MatrixGenerator<scalar>::n;
    uint k = sigma.size();

    OrthonormalEntriesMatrixGenerator<scalar> u_gen(
        m, k, MatrixGenerator<scalar>::gen());
    OrthonormalEntriesMatrixGenerator<scalar> v_gen(
        n, k, MatrixGenerator<scalar>::gen());

    return u_gen.generateMatrix() * sigma.asDiagonal() *
           v_gen.generateMatrix().adjoint();
}

// NearRankOneMatrixGenerator class

template <typename scalar>
Eigen::VectorX<scalar>
NearRankOneMatrixGenerator<scalar>::getSigma(uint m, uint n, scalar eps) {

    Eigen::VectorX<scalar> sigma =
        Eigen::VectorX<scalar>::Constant(std::min(m, n), eps);
    sigma(0) = 1;
    return sigma;
}

template <typename scalar>
NearRankOneMatrixGenerator<scalar>::NearRankOneMatrixGenerator(uint m, uint n,
                                                               scalar eps)
    : SigmaMatrixGenerator<scalar>(m, n, getSigma(m, n, eps)) {}

template <typename scalar>
NearRankOneMatrixGenerator<scalar>::NearRankOneMatrixGenerator(uint m, uint n,
                                                               scalar eps,
                                                               int seed)
    : SigmaMatrixGenerator<scalar>(m, n, seed, getSigma(m, n, eps)) {}

// NearSingularMatrixGenerator class

template <typename scalar>
Eigen::VectorX<scalar>
NearSingularMatrixGenerator<scalar>::getSigma(uint m, uint n, scalar eps) {

    Eigen::VectorX<scalar> sigma =
        Eigen::VectorX<scalar>::Constant(std::min(m, n), 1);
    sigma(std::min(m, n) - 1) = eps;
    return sigma;
}

template <typename scalar>
NearSingularMatrixGenerator<scalar>::NearSingularMatrixGenerator(uint m, uint n,
                                                               scalar eps)
    : SigmaMatrixGenerator<scalar>(m, n, getSigma(m, n, eps)) {}

template <typename scalar>
NearSingularMatrixGenerator<scalar>::NearSingularMatrixGenerator(uint m, uint n,
                                                               scalar eps,
                                                               int seed)
    : SigmaMatrixGenerator<scalar>(m, n, seed, getSigma(m, n, eps)) {}

} // namespace SubsetSelection
