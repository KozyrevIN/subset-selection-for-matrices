#include <iostream>

#include <Eigen/Core>

#include "../include/SigmaMatrixGenerator.h"

int main(int argc, char *argv[]) {
    Eigen::VectorX<double> S(2);
    S << 1, 2;
    std::cout << S << std::endl;
    MatSubset::Bench::SigmaMatrixGenerator<double> mat_gen(2, 3, S);
    Eigen::MatrixX<double> mat = mat_gen.generateMatrix();
    std::cout << mat_gen.getMatrixType() << std::endl;
    std::cout << mat << std::endl;

    return 0;
}