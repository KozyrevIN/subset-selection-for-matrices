#include <iostream>

#include "subset_selection/include/subset_selection.h"

using namespace SubsetSelection;

int main()
{
    auto mat_gen_1 = new OrthonormalVectorsMatrixGenerator<double>(10, 50);
    //auto mat_gen_2 = new WeightedGraphIncidenceMatrixGenerator<double>(100, 500);
    
    SpectralSelectionSelector<double> selector_1;
    DualSetSelector<double> selector_2;
    SpectralRemovalSelector<double> selector_3;
    SubsetSelector<double> selector_4;
    InterlacingFamiliesSelector<double> selector_5;

    auto alg_list = std::vector<SubsetSelector<double>*>();
    alg_list.push_back(&selector_1);
    alg_list.push_back(&selector_2);
    alg_list.push_back(&selector_3);
    alg_list.push_back(&selector_4);
    alg_list.push_back(&selector_5);

    Tester<double> t;
    //std::cout << t.testAlgorithmsOnMatrix(mat_gen, alg_list, 500, 1);
    //t.scatterPoints<SubsetSelection::Norm::L2>(mat_gen_1, alg_list, 100, 500, 4, 16);
    //t.scatterPoints<SubsetSelection::Norm::L2>(mat_gen_1, alg_list, 100, 200, 1, 32);

    //t.scatterPoints<SubsetSelection::Norm::L2>(mat_gen_2, alg_list, 100, 500, 4, 16);
    //t.scatterPoints<SubsetSelection::Norm::L2>(mat_gen_2, alg_list, 100, 200, 1, 32);
    //t.scatterPoints<SubsetSelection::Norm::L2>(mat_gen_2, alg_list, 100, 200, 1, 32);

    std::cout << t.testAlgorithmsOnMatrix(mat_gen_1, alg_list, 20, 1);

    delete mat_gen_1;
    //delete mat_gen_2;
    return 0;
}