#include <iostream>

#include "subset_selection/include/subset_selection.h"

using namespace SubsetSelection;

int main()
{
    auto mat_gen = new type3MatrixGenerator<double>(10, 100, 1);
    //auto A = mat_gen.generateMatrix();
    SpectralSelectionSelector<double> selector_1;
    //VolumeRemovalSelector<double> selector_2;
    //DualSetSelector<double> selector_3;
    //FrobeniusRemovalSelector<double> selector_4;
    SubsetSelector<double> selector_5("random");
    InterlacingFamiliesSelector<double> selector_6;

    auto alg_list = std::vector<SubsetSelector<double>*>();
    alg_list.push_back(&selector_1);
    //alg_list.push_back(&selector_2);
    //alg_list.push_back(&selector_3);
    //alg_list.push_back(&selector_4);
    alg_list.push_back(&selector_5);
    alg_list.push_back(&selector_6);

    Tester<double> t;
    std::cout << t.testAlgorithmsOnMatrix(mat_gen, alg_list, 10, 2);
    //t.scatterPoints<Norm::L2>(mat_gen, &selector_1, 100);
    //t.scatterPoints<Norm::L2>(mat_gen, &selector_2, 100);
    //t.scatterPoints<Norm::L2>(mat_gen, &selector_3, 100);
    delete mat_gen;
}