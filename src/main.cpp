#include <iostream>

#include "../include/tester.h"
#include "../include/subset_selection_algorithms/all_algorithms.h"


int main()
{
    Eigen::setNbThreads(8);
    Eigen::initParallel();

    auto mat_gen = type3MatrixGenerator<double>(30, 3000, 0.1);
    auto A = mat_gen.generateMatrix();
    SubsetSelector<double> selector_1("bobs");
    RankRevealingQRSelector<double> selector_2;
    GreedyRemoval1Selector<double> selector_3(1e-8);
    GreedyRemoval2Selector<double> selector_4;

    auto alg_list = std::vector<SubsetSelector<double>*>();
    alg_list.push_back(&selector_1);
    alg_list.push_back(&selector_2);
    alg_list.push_back(&selector_3);
    alg_list.push_back(&selector_4);

    Tester<double> t;
    std::cout << t.testAlgorithmsOnMatrix(A, alg_list, 30);
}