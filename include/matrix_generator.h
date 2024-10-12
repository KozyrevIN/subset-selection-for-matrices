#include <eigen3/Eigen/Dense>

/*
Базовый класс генератора матриц
*/
template <typename scalar>
class MatrixGenerator {
protected:
    uint m;
    uint n;
    
public:
    MatrixGenerator(uint m, uint n);
    virtual Eigen::MatrixX<scalar> generateMatrix(uint seed);
    virtual Eigen::MatrixX<scalar> generateMatrix();
};

/*
Генератор унитарной матрицы заданых размеров
*/
template <typename scalar>
class UnitaryMatrixGenerator : public MatrixGenerator<scalar> {
public:
    UnitaryMatrixGenerator(uint m, uint n);
    Eigen::MatrixX<scalar> generateMatrix(uint seed) override;
    Eigen::MatrixX<scalar> generateMatrix() override;
};

/*
Генератор матрицы заданных размеров с задаными набором сингулярных чисел
*/
template <typename scalar>
class SigmaMatrixGenerator : public MatrixGenerator<scalar> {
public:
    SigmaMatrixGenerator(uint m, uint n);
    Eigen::MatrixX<scalar> generateMatrixWithSigma(const Eigen::VectorX<scalar>& sigma, uint seed_1, uint seed_2);
    Eigen::MatrixX<scalar> generateMatrixWithSigma(const Eigen::VectorX<scalar>& sigma);
};

/*
Генератор матрицы со всеми сингулярными числами 1
*/
template <typename scalar>
class type1MatrixGenerator : public SigmaMatrixGenerator<scalar> {
private:
    scalar eps;

public:
    type1MatrixGenerator(uint m, uint n, scalar eps);
    Eigen::MatrixX<scalar> generateMatrix(uint seed_1, uint seed_2);
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
    Eigen::MatrixX<scalar> generateMatrix(uint seed_1, uint seed_2);
    Eigen::MatrixX<scalar> generateMatrix() override;
};

/*
Генератор матрицы со всеми сингулярными числами кроме последнего 1, последнее - эпсилон
*/
template <typename scalar>
class type3MatrixGenerator : public SigmaMatrixGenerator<scalar> {
private:
    scalar eps;
    
public:
    type3MatrixGenerator(uint m, uint n, scalar eps);
    Eigen::MatrixX<scalar> generateMatrix(uint seed_1, uint seed_2);
    Eigen::MatrixX<scalar> generateMatrix() override;
};
