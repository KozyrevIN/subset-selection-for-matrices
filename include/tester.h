#include <string>
#include <vector>

#ifndef matrix_generator_header
    #include "matrix_generator.h"
    #define matrix_generator_header
#endif

#ifndef subset_selector_header
#include "subset_selection_algorithms/subset_selector.h"
    #define subset_selector_header
#endif

template<typename scalar>
class Tester 
{
public:
    Tester();

    std::string testAlgorithmOnMatrix(const Eigen::MatrixX<scalar>& A, SubsetSelector<scalar>* algorithm, uint k);
    std::string testAlgorithmsOnMatrix(const Eigen::MatrixX<scalar>& A, std::vector<SubsetSelector<scalar>*> algorithms, uint k);
    std::string testAlgorithmOnMatrix(MatrixGenerator<scalar>* mat_gen, SubsetSelector<scalar>* algorithm, uint k);
    std::string testAlgorithmOnMatrix(MatrixGenerator<scalar>* mat_gen, SubsetSelector<scalar>* algorithm, uint k, uint cycles);
    std::string testAlgorithmsOnMatrix(MatrixGenerator<scalar>* mat_gen, std::vector<SubsetSelector<scalar>*> algorithms, uint k, uint cycles);
};