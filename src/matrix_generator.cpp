#include <random>
#include <cassert>
#include <iostream>

#include "../include/matrix_generator.h"

/*
Базовый класс генератора матриц
*/
template <typename scalar>
MatrixGenerator<scalar>::MatrixGenerator(uint m, uint n) : m(m), n(n) {
    //do nothing
}

template <typename scalar>
Eigen::MatrixX<scalar> MatrixGenerator<scalar>::generateMatrix(uint seed) {
    return Eigen::MatrixX<scalar>(m, n);
}

template <typename scalar>
Eigen::MatrixX<scalar> MatrixGenerator<scalar>::generateMatrix() {
    std::random_device rd;
    return generateMatrix(rd());
}

template class MatrixGenerator<float>;
template class MatrixGenerator<double>;

/*
Генератор унитарной матрицы заданных размеров, m >= n
*/
template <typename scalar>
UnitaryMatrixGenerator<scalar>::UnitaryMatrixGenerator(uint m, uint n) : MatrixGenerator<scalar>(m, n) {
   assert(m >= n && "Invalid matrix size, m must be larger then n");
}

template <typename scalar>
Eigen::MatrixX<scalar> UnitaryMatrixGenerator<scalar>::generateMatrix(uint seed) {
    uint& m = MatrixGenerator<scalar>::m;
    uint& n = MatrixGenerator<scalar>::n;
    std::mt19937 gen(seed);
    std::normal_distribution<scalar> dis(0.0, 1.0);

    Eigen::MatrixX<scalar> tmp(m, n);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            tmp(i, j) = dis(gen);
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

template <typename scalar>
Eigen::MatrixX<scalar> UnitaryMatrixGenerator<scalar>::generateMatrix() {
    std::random_device rd;
    return generateMatrix(rd());
}

template class UnitaryMatrixGenerator<float>;
template class UnitaryMatrixGenerator<double>;

/*
Генератор матрицы заданных размеров с заданными набором сингулярных чисел
*/
template <typename scalar>
SigmaMatrixGenerator<scalar>::SigmaMatrixGenerator(uint m, uint n) : MatrixGenerator<scalar>(m , n) {
    //do nothing
}

template <typename scalar>
Eigen::MatrixX<scalar> SigmaMatrixGenerator<scalar>::generateMatrixWithSigma(const Eigen::VectorX<scalar>& sigma, uint seed_1, uint seed_2) {
    uint& m = MatrixGenerator<scalar>::m;
    uint& n = MatrixGenerator<scalar>::n;
    uint k = sigma.rows();

    assert((m >= k) && (n >= k) && "Invalid combination of matrix size and number of singular values, n and m must be larger then k");

    UnitaryMatrixGenerator<scalar> u_gen(m, k); 
    UnitaryMatrixGenerator<scalar> v_gen(n, k); 

    return u_gen.generateMatrix(seed_1) * sigma.asDiagonal() * v_gen.generateMatrix(seed_2).adjoint();
}

template <typename scalar>
Eigen::MatrixX<scalar> generateMatrixWithSigma(const Eigen::VectorX<scalar>& sigma) {
    std::random_device rd_1, rd_2;
    return generateMatrixWithSigma<scalar>(sigma, rd_1(), rd_2());
}

template class SigmaMatrixGenerator<float>;
template class SigmaMatrixGenerator<double>;

/*
Генератор матрицы со всеми сингулярными числами 1
*/
template <typename scalar>
type1MatrixGenerator<scalar>::type1MatrixGenerator(uint m, uint n, scalar eps) : SigmaMatrixGenerator<scalar>(m, n), eps(eps) {
    //do nothing
}

template <typename scalar>
Eigen::MatrixX<scalar> type1MatrixGenerator<scalar>::generateMatrix(uint seed_1, uint seed_2) {
    uint& m = MatrixGenerator<scalar>::m;
    uint& n = MatrixGenerator<scalar>::n;

    Eigen::VectorX<scalar> sigma(m);
    for (uint i = 0; i < m; ++i) {
        sigma(i) = 1;
    }

    return SigmaMatrixGenerator<scalar>::generateMatrixWithSigma(sigma, seed_1, seed_2);
};

template <typename scalar>
Eigen::MatrixX<scalar> type1MatrixGenerator<scalar>::generateMatrix() {
    std::random_device rd_1, rd_2;
    return generateMatrix(rd_1(), rd_2());
}

template class type1MatrixGenerator<float>;
template class type1MatrixGenerator<double>;

/*
Генератор матрицы с 1 сингулярными числом 1 и остальными эпсилон
*/
template <typename scalar>
type2MatrixGenerator<scalar>::type2MatrixGenerator(uint m, uint n, scalar eps) : SigmaMatrixGenerator<scalar>(m, n), eps(eps) {
    //do nothing
}

template <typename scalar>
Eigen::MatrixX<scalar> type2MatrixGenerator<scalar>::generateMatrix(uint seed_1, uint seed_2) {
    uint& m = MatrixGenerator<scalar>::m;
    uint& n = MatrixGenerator<scalar>::n;

    Eigen::VectorX<scalar> sigma(m);
    sigma(0) = 1;
    for (uint i = 1; i < m; ++i) {
        sigma(i) = eps;
    }

    return SigmaMatrixGenerator<scalar>::generateMatrixWithSigma(sigma, seed_1, seed_2);
};

template <typename scalar>
Eigen::MatrixX<scalar> type2MatrixGenerator<scalar>::generateMatrix() {
    std::random_device rd_1, rd_2;
    return generateMatrix(rd_1(), rd_2());
}

template class type2MatrixGenerator<float>;
template class type2MatrixGenerator<double>;

/*
Генератор матрицы со всеми сингулярными числами кроме посдеднего 1, последнее - эпсилон
*/
template <typename scalar>
type3MatrixGenerator<scalar>::type3MatrixGenerator(uint m, uint n, scalar eps) : SigmaMatrixGenerator<scalar>(m, n), eps(eps) {
    //do nothing
}

template <typename scalar>
Eigen::MatrixX<scalar> type3MatrixGenerator<scalar>::generateMatrix(uint seed_1, uint seed_2) {
    uint& m = MatrixGenerator<scalar>::m;
    uint& n = MatrixGenerator<scalar>::n;
    Eigen::VectorX<scalar> sigma(m);
    for (uint i = 0; i < m - 1; ++i) {
        sigma(i) = 1;
    }
    sigma(m - 1) = eps;

    return SigmaMatrixGenerator<scalar>::generateMatrixWithSigma(sigma, seed_1, seed_2);
};

template <typename scalar>
Eigen::MatrixX<scalar> type3MatrixGenerator<scalar>::generateMatrix() {
    std::random_device rd_1, rd_2;
    return generateMatrix(rd_1(), rd_2());
}

template class type3MatrixGenerator<float>;
template class type3MatrixGenerator<double>;