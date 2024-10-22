#include <iostream>

#include "subset_selection/include/subset_selection.h"

using namespace SubsetSelection;

int main()
{
    auto mat_gen = new type3MatrixGenerator<double>(20, 300, 0.1);
    //auto A = mat_gen.generateMatrix();
    GreedySelection1Selector<double> selector_1;
    GreedyRemoval2Selector<double> selector_2;
    SubsetSelector<double> selector_3("random");

    auto alg_list = std::vector<SubsetSelector<double>*>();
    //alg_list.push_back(&selector_1);
    alg_list.push_back(&selector_2);
    alg_list.push_back(&selector_3);

    Tester<double> t;
    std::cout << t.testAlgorithmsOnMatrix(mat_gen, alg_list, 50, 10);
    t.scatterPoints<Norm::Frobenius>(mat_gen, &selector_3, 1000);
    delete mat_gen;
}