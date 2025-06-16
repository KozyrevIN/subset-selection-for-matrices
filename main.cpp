#include <iostream>

#include "subset_selection/include/subset_selection.h"

using namespace SubsetSelection;

int main()
{
    const int m = 100;
    const int n = 5000;
    const int mpp = 32;
    auto gaussian = new GaussianMatrixGenerator<double>(m, n);
    auto orthonormal = new OrthonormalVectorsMatrixGenerator<double>(m, n);
    auto random_graph = new WeightedGraphIncidenceMatrixGenerator<double>(m, n);
    
    SpectralSelectionSelector<double> selector_1;
    VolumeRemovalSelector<double> selector_2;
    FrobeniusRemovalSelector<double> selector_3;
    SubsetSelector<double> selector_4;
    //InterlacingFamiliesSelector<double> selector_5;

    auto alg_list = std::vector<SubsetSelector<double>*>();
    alg_list.push_back(&selector_1);
    alg_list.push_back(&selector_2);
    alg_list.push_back(&selector_3);
    alg_list.push_back(&selector_4);
    //alg_list.push_back(&selector_5);

    Tester<double> t;
    t.scatterPoints<SubsetSelection::Norm::Frobenius>(gaussian, alg_list, m, n, (n - m)/10, mpp);
    t.scatterPoints<SubsetSelection::Norm::Frobenius>(gaussian, alg_list, m, 2*m, 1, mpp);

    t.scatterPoints<SubsetSelection::Norm::Frobenius>(orthonormal, alg_list, m, n, (n - m)/10, mpp);
    t.scatterPoints<SubsetSelection::Norm::Frobenius>(orthonormal, alg_list, m, 2*m, 1, mpp);

    t.scatterPoints<SubsetSelection::Norm::Frobenius>(random_graph, alg_list, m, n, (n - m)/10, mpp);
    t.scatterPoints<SubsetSelection::Norm::Frobenius>(random_graph, alg_list, m, 2*m, 1, mpp);

    t.scatterPoints<SubsetSelection::Norm::Frobenius>(random_graph, alg_list, m, m + 1, 1, 1);

    delete gaussian;
    delete orthonormal;
    delete random_graph;
    return 0;
}