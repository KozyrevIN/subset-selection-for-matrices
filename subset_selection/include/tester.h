#ifndef TESTER_H
#define TESTER_H

#include <string>
#include <vector>

#include "matrix_generator.h"
#include "subset_selection_algorithms/subset_selector.h"

namespace SubsetSelection
{

template <typename scalar>
class Tester 
{
public:
    Tester();

    std::string testAlgorithmOnMatrix(const Eigen::MatrixX<scalar>& A, SubsetSelector<scalar>* algorithm, uint k);
    std::string testAlgorithmsOnMatrix(const Eigen::MatrixX<scalar>& A, std::vector<SubsetSelector<scalar>*> algorithms, uint k);

    std::string testAlgorithmOnMatrix(MatrixGenerator<scalar>* mat_gen, SubsetSelector<scalar>* algorithm, uint k);
    std::string testAlgorithmOnMatrix(MatrixGenerator<scalar>* mat_gen, SubsetSelector<scalar>* algorithm, uint k, uint cycles);
    std::string testAlgorithmsOnMatrix(MatrixGenerator<scalar>* mat_gen, std::vector<SubsetSelector<scalar>*> algorithms, uint k, uint cycles);

    template <Norm norm>
    void scatterPoints(MatrixGenerator<scalar>* mat_gen, SubsetSelector<scalar>* algorithm, uint points_for_k);
};

}

#include "../src/tester.hpp"

#endif