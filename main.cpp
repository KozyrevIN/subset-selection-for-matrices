#include <iostream>

#include "subset_selection/include/subset_selection.h"

using namespace SubsetSelection;

int main()
{
    const int m = 50;
    const int n = 1000;
    const int mpp = 32;
    auto gaussian = new GaussianMatrixGenerator<double>(m, n);
    auto orthonormal = new OrthonormalVectorsMatrixGenerator<double>(m, n);
    auto random_graph = new WeightedGraphIncidenceMatrixGenerator<double>(m, n);
    
    SubsetSelector<double> selector_1;
    SpectralSelectionSelector<double> selector_2;
    FrobeniusRemovalSelector<double> selector_3;
    SpectralRemovalSelector<double> selector_4;
    DualSetSelector<double> selector_5;
    //InterlacingFamiliesSelector<double> selector_5;

    auto alg_list = std::vector<SubsetSelector<double>*>();
    alg_list.push_back(&selector_5);
    alg_list.push_back(&selector_2);

    Tester<double> t;
    //gaussian matrices
    /*
    t.scatterPoints<SubsetSelection::Norm::Frobenius>(gaussian, alg_list, m, n, (n - m)/m, mpp);
    t.scatterPoints<SubsetSelection::Norm::Frobenius>(gaussian, alg_list, m, 2*m, 1, mpp);

    std::cout << "25%" << std::endl;

    t.scatterPoints<SubsetSelection::Norm::L2>(gaussian, alg_list, m, n, (n - m)/m, mpp);
    t.scatterPoints<SubsetSelection::Norm::L2>(gaussian, alg_list, m, 2*m, 1, mpp);

    std::cout << "50%" << std::endl;

    //weighted graph matrices

    t.scatterPoints<SubsetSelection::Norm::Frobenius>(random_graph, alg_list, m, n, (n - m)/m, mpp);
    t.scatterPoints<SubsetSelection::Norm::Frobenius>(random_graph, alg_list, m, 2*m, 1, mpp);

    std::cout << "75%" << std::endl;
    alg_list.push_back(&selector_1);
    alg_list.push_back(&selector_3);
    alg_list.push_back(&selector_4);
    */

    t.scatterPoints<SubsetSelection::Norm::L2>(random_graph, alg_list, m, n, (n - m)/m, mpp);
    t.scatterPoints<SubsetSelection::Norm::L2>(random_graph, alg_list, m, 2*m, 1, mpp);

    t.scatterPoints<SubsetSelection::Norm::L2>(random_graph, alg_list, m, m + 1, 1, 1);

    //std::cout << t.testAlgorithmsOnMatrix(random_graph, alg_list, 50, mpp);

    delete gaussian;
    delete orthonormal;
    delete random_graph;
    return 0;
}