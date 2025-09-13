#include <iostream>

#include <Eigen/Core>

#include "../include/OrthonormalVectorsMatrixGenerator.h"

int main(int argc, char *argv[]) {
    MatSubset::Bench::OrthonormalVectorsMatrixGenerator<double> mat_gen(2, 3);
    Eigen::MatrixX<double> mat = mat_gen.generateMatrix();
    std::cout << mat_gen.getMatrixType() << std::endl;
    std::cout << mat * mat.transpose() << std::endl;

    return 0;
}