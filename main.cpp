#include <iostream>

#include "subset_selection/include/subset_selection.h"

using namespace SubsetSelection;

int main()
{
    auto mat_gen = new NearRankOneMatrixGenerator<double>(20, 200, 0.1);
    //auto A = mat_gen.generateMatrix();
    SpectralSelectionSelector<double> selector_1;
    //VolumeRemovalSelector<double> selector_2;
    DualSetSelector<double> selector_3;
    FrobeniusRemovalSelector<double> selector_4;
    SubsetSelector<double> selector_5;
    //InterlacingFamiliesSelector<double> selector_6;

    auto alg_list = std::vector<SubsetSelector<double>*>();
    alg_list.push_back(&selector_1);
    //alg_list.push_back(&selector_2);
    alg_list.push_back(&selector_3);
    alg_list.push_back(&selector_4);
    alg_list.push_back(&selector_5);
    //alg_list.push_back(&selector_6);

    Tester<double> t;
    t.scatterPoints<SubsetSelection::Norm::L2>(mat_gen, alg_list, 20, 200, 1);

    delete mat_gen;
}