#ifndef MATRIX_GENERATOR_H
#define MATRIX_GENERATOR_H

#include <eigen3/Eigen/Dense>
#include <random>

namespace SubsetSelection {

/*
Базовый класс генератора матриц
*/
template <typename scalar> class MatrixGenerator {
  protected:
    uint m;
    uint n;
    std::mt19937 gen;

  public:
    MatrixGenerator(uint m, uint n);
    MatrixGenerator(uint m, uint n, int seed);
    std::pair<uint, uint> getMatrixSize();
    virtual Eigen::MatrixX<scalar> generateMatrix();
};

/*
Генератор унитарной матрицы заданых размеров
*/
template <typename scalar>
class UnitaryMatrixGenerator : public MatrixGenerator<scalar> {
  public:
    UnitaryMatrixGenerator(uint m, uint n);
    UnitaryMatrixGenerator(uint m, uint n, int seed);
    Eigen::MatrixX<scalar> generateMatrix() override;
};

/*
Генератор матрицы заданных размеров с задаными набором сингулярных чисел
*/
template <typename scalar>
class SigmaMatrixGenerator : public MatrixGenerator<scalar> {
  public:
    SigmaMatrixGenerator(uint m, uint n);
    SigmaMatrixGenerator(uint m, uint n, int seed);
    Eigen::MatrixX<scalar>
    generateMatrixWithSigma(const Eigen::VectorX<scalar> &sigma);
};

/*
Генератор матрицы со всеми сингулярными числами 1
*/
template <typename scalar>
class type1MatrixGenerator : public SigmaMatrixGenerator<scalar> {
  public:
    type1MatrixGenerator(uint m, uint n);
    type1MatrixGenerator(uint m, uint n, int seed);
    Eigen::MatrixX<scalar> generateMatrix() override;
};

/*
Генератор матрицы с 1 сингулярными числом 1 и остальными эпсилон
*/
template <typename scalar>
class type2MatrixGenerator : public SigmaMatrixGenerator<scalar> {
  private:
    scalar eps;

  public:
    type2MatrixGenerator(uint m, uint n, scalar eps);
    type2MatrixGenerator(uint m, uint n, scalar eps, int seed);
    Eigen::MatrixX<scalar> generateMatrix() override;
};

/*
Генератор матрицы со всеми сингулярными числами кроме последнего 1, последнее -
эпсилон
*/
template <typename scalar>
class type3MatrixGenerator : public SigmaMatrixGenerator<scalar> {
  private:
    scalar eps;

  public:
    type3MatrixGenerator(uint m, uint n, scalar eps);
    type3MatrixGenerator(uint m, uint n, scalar eps, int seed);
    Eigen::MatrixX<scalar> generateMatrix() override;
};

} // namespace SubsetSelection

#include "../src/matrix_generator.hpp"

#endif
