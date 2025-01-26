#ifndef MATRIX_GENERATOR_H
#define MATRIX_GENERATOR_H

#include <eigen3/Eigen/Dense>
#include <random>

namespace SubsetSelection {

/*
Base matrix generator class
*/
template <typename scalar> class MatrixGenerator {
  protected:
    uint m;
    uint n;
    std::mt19937 gen;

  public:
    MatrixGenerator(uint m, uint n);
    MatrixGenerator(uint m, uint n, int seed);
    std::pair<uint, uint> getMatrixSize();
    virtual Eigen::MatrixX<scalar> generateMatrix();
};

/*
Generator of a random matrix with orthonormal columns/rows
(depending on which of m and n is larger)
*/
template <typename scalar>
class OrthonormalEntriesMatrixGenerator : public MatrixGenerator<scalar> {
  public:
    OrthonormalEntriesMatrixGenerator(uint m, uint n);
    OrthonormalEntriesMatrixGenerator(uint m, uint n, int seed);
    Eigen::MatrixX<scalar> generateMatrix() override;
};

/*
Generator of a random matrix with a given set of singular values
*/
template <typename scalar>
class SigmaMatrixGenerator : public MatrixGenerator<scalar> {
  private:
    Eigen::VectorX<scalar> sigma;

  public:
    SigmaMatrixGenerator(uint m, uint n, const Eigen::VectorX<scalar> &sigma);
    SigmaMatrixGenerator(uint m, uint n, int seed,
                         const Eigen::VectorX<scalar> &sigma);
    Eigen::MatrixX<scalar> generateMatrix() override;
};

/*
Generator of a random matrix with the first singular value equal to 1
and other to eps (nearly rank-one matrix)
*/
template <typename scalar>
class NearRankOneMatrixGenerator : public SigmaMatrixGenerator<scalar> {
  private:
    Eigen::VectorX<scalar> getSigma(uint m, uint n, scalar eps);

  public:
    NearRankOneMatrixGenerator(uint m, uint n, scalar eps);
    NearRankOneMatrixGenerator(uint m, uint n, scalar eps, int seed);
};

/*
Generator of a random matrix with the first min(m, n) - 1 singular
values equal to 1 and the remaining one equal to eps (matrix being near to
singular)
*/
template <typename scalar>
class NearSingularMatrixGenerator : public SigmaMatrixGenerator<scalar> {
  private:
    Eigen::VectorX<scalar> getSigma(uint m, uint n, scalar eps);

  public:
    NearSingularMatrixGenerator(uint m, uint n, scalar eps);
    NearSingularMatrixGenerator(uint m, uint n, scalar eps, int seed);
};

/*
Generator of a matrix of right singular vectors of an edge-vertex incidence
matrix of a random fully-connected graph with m verticies and n edges.
*/
template <typename scalar>
class GraphIncidenceMatrixGenerator : public MatrixGenerator<scalar> {
  private:
    std::vector<std::pair<uint, uint>> randomEdgeList();

    bool checkConnectivity(const std::vector<std::pair<uint, uint>> &edge_list);

    Eigen::MatrixX<scalar>
    incidenceMatrix(const std::vector<std::pair<uint, uint>> &edge_list);

  public:
    GraphIncidenceMatrixGenerator(uint m, uint n);
    GraphIncidenceMatrixGenerator(uint m, uint n, int seed);
    Eigen::MatrixX<scalar> generateMatrix() override;
};

} // namespace SubsetSelection

#include "../src/matrix_generator.hpp"

#endif
