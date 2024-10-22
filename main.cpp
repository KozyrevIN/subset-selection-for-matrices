#include <iostream>

#include "subset_selection/include/subset_selection.h"

using namespace SubsetSelection;

int main()
{
    auto mat_gen = new type1MatrixGenerator<double>(20, 300);
    //auto A = mat_gen.generateMatrix();
    GreedySelection1Selector<double> selector_1;
    GreedyRemoval2Selector<double> selector_2;

    auto alg_list = std::vector<SubsetSelector<double>*>();
    alg_list.push_back(&selector_1);
    alg_list.push_back(&selector_2);

    Tester<double> t;
    //std::cout << t.testAlgorithmsOnMatrix(mat_gen, alg_list, 51, 10);
    t.scatterPoints<Norm::Frobenius>(mat_gen, &selector_2, 10);
    delete mat_gen;
}