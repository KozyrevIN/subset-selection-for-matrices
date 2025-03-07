#include <iostream>

#include "subset_selection/include/subset_selection.h"

using namespace SubsetSelection;

int main()
{
    auto mat_gen = new WeightedGraphIncidenceMatrixGenerator<double>(100, 5000);
    SpectralSelectionSelector<double> selector_1;
    DualSetSelector<double> selector_2;
    SpectralRemovalSelector<double> selector_3;
    SubsetSelector<double> selector_4;
    RankRevealingQRSelector<double> selector_5;

    auto alg_list = std::vector<SubsetSelector<double>*>();
    alg_list.push_back(&selector_1);
    alg_list.push_back(&selector_2);
    alg_list.push_back(&selector_3);
    alg_list.push_back(&selector_4);
    //alg_list.push_back(&selector_5);

    Tester<double> t;
    //std::cout << t.testAlgorithmsOnMatrix(mat_gen, alg_list, 500, 1);
    t.scatterPoints<SubsetSelection::Norm::L2>(mat_gen, alg_list, 100, 5000, 49, 4);
    t.scatterPoints<SubsetSelection::Norm::L2>(mat_gen, alg_list, 100, 200, 1, 32);

    delete mat_gen;
}