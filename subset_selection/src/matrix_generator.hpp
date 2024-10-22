#include <cassert>
#include <iostream>

namespace SubsetSelection
{

/*
Базовый класс генератора матриц
*/
template <typename scalar>
MatrixGenerator<scalar>::MatrixGenerator(uint m, uint n) : m(m), n(n) {
    std::random_device rd;
    gen = std::mt19937(rd());
}

template <typename scalar>
MatrixGenerator<scalar>::MatrixGenerator(uint m, uint n, int seed) : m(m), n(n) {
    gen = std::mt19937(seed);
}

template <typename scalar>
std::pair <uint, uint> MatrixGenerator<scalar>::getMatrixSize() {
    return std::make_pair(m, n);
}

template <typename scalar>
Eigen::MatrixX<scalar> MatrixGenerator<scalar>::generateMatrix() {
    return Eigen::MatrixX<scalar>(m, n);
}

/*
Генератор унитарной матрицы заданных размеров, m >= n
*/
template <typename scalar>
UnitaryMatrixGenerator<scalar>::UnitaryMatrixGenerator(uint m, uint n) : MatrixGenerator<scalar>(m, n) {
   assert(m >= n && "Invalid matrix size, m must be larger then n");
}

template <typename scalar>
UnitaryMatrixGenerator<scalar>::UnitaryMatrixGenerator(uint m, uint n, int seed) : MatrixGenerator<scalar>(m, n, seed) {
   assert(m >= n && "Invalid matrix size, m must be larger then n");
}

template <typename scalar>
Eigen::MatrixX<scalar> UnitaryMatrixGenerator<scalar>::generateMatrix() {
    uint& m = MatrixGenerator<scalar>::m;
    uint& n = MatrixGenerator<scalar>::n;
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

    for(int i = 0; i < n; ++i) {
        if (std::abs(Rdiag(i) != 0)) {
            Q.row(i) *= Rdiag(i) / std::abs(Rdiag(i));
        }
    }

    return Q;
}

/*
Генератор матрицы заданных размеров с заданными набором сингулярных чисел
*/
template <typename scalar>
SigmaMatrixGenerator<scalar>::SigmaMatrixGenerator(uint m, uint n) : MatrixGenerator<scalar>(m , n) {
    //do nothing
}

template <typename scalar>
SigmaMatrixGenerator<scalar>::SigmaMatrixGenerator(uint m, uint n, int seed) : MatrixGenerator<scalar>(m , n, seed) {
    //do nothing
}

template <typename scalar>
Eigen::MatrixX<scalar> SigmaMatrixGenerator<scalar>::generateMatrixWithSigma(const Eigen::VectorX<scalar>& sigma) {
    uint& m = MatrixGenerator<scalar>::m;
    uint& n = MatrixGenerator<scalar>::n;
    uint k = sigma.rows();

    assert((m >= k) && (n >= k) && "Invalid combination of matrix size and number of singular values, n and m must be larger then k");

    UnitaryMatrixGenerator<scalar> u_gen(m, k, MatrixGenerator<scalar>::gen()); 
    UnitaryMatrixGenerator<scalar> v_gen(n, k, MatrixGenerator<scalar>::gen()); 

    return u_gen.generateMatrix() * sigma.asDiagonal() * v_gen.generateMatrix().adjoint();
}

/*
Генератор матрицы со всеми сингулярными числами 1
*/
template <typename scalar>
type1MatrixGenerator<scalar>::type1MatrixGenerator(uint m, uint n) : SigmaMatrixGenerator<scalar>(m, n) {
    //do nothing
}

template <typename scalar>
type1MatrixGenerator<scalar>::type1MatrixGenerator(uint m, uint n, int seed) : SigmaMatrixGenerator<scalar>(m, n, seed) {
    //do nothing
}

template <typename scalar>
Eigen::MatrixX<scalar> type1MatrixGenerator<scalar>::generateMatrix() {
    uint& m = MatrixGenerator<scalar>::m;
    uint& n = MatrixGenerator<scalar>::n;

    Eigen::VectorX<scalar> sigma(m);
    for (uint i = 0; i < m; ++i) {
        sigma(i) = 1;
    }

    return SigmaMatrixGenerator<scalar>::generateMatrixWithSigma(sigma);
};

/*
Генератор матрицы с 1 сингулярными числом 1 и остальными эпсилон
*/
template <typename scalar>
type2MatrixGenerator<scalar>::type2MatrixGenerator(uint m, uint n, scalar eps) : SigmaMatrixGenerator<scalar>(m, n), eps(eps) {
    //do nothing
}

template <typename scalar>
type2MatrixGenerator<scalar>::type2MatrixGenerator(uint m, uint n, scalar eps, int seed) : SigmaMatrixGenerator<scalar>(m, n, seed), eps(eps) {
    //do nothing
}

template <typename scalar>
Eigen::MatrixX<scalar> type2MatrixGenerator<scalar>::generateMatrix() {
    uint& m = MatrixGenerator<scalar>::m;
    uint& n = MatrixGenerator<scalar>::n;

    Eigen::VectorX<scalar> sigma(m);
    sigma(0) = 1;
    for (uint i = 1; i < m; ++i) {
        sigma(i) = eps;
    }

    return SigmaMatrixGenerator<scalar>::generateMatrixWithSigma(sigma);
}

/*
Генератор матрицы со всеми сингулярными числами кроме посдеднего 1, последнее - эпсилон
*/
template <typename scalar>
type3MatrixGenerator<scalar>::type3MatrixGenerator(uint m, uint n, scalar eps) : SigmaMatrixGenerator<scalar>(m, n), eps(eps) {
    //do nothing
}

template <typename scalar>
type3MatrixGenerator<scalar>::type3MatrixGenerator(uint m, uint n, scalar eps, int seed) : SigmaMatrixGenerator<scalar>(m, n, seed), eps(eps) {
    //do nothing
}

template <typename scalar>
Eigen::MatrixX<scalar> type3MatrixGenerator<scalar>::generateMatrix() {
    uint& m = MatrixGenerator<scalar>::m;
    uint& n = MatrixGenerator<scalar>::n;
    Eigen::VectorX<scalar> sigma(m);
    for (uint i = 0; i < m - 1; ++i) {
        sigma(i) = 1;
    }
    sigma(m - 1) = eps;

    return SigmaMatrixGenerator<scalar>::generateMatrixWithSigma(sigma);
}

}
