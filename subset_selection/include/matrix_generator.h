#ifndef MATRIX_GENERATOR_H
#define MATRIX_GENERATOR_H

#include <eigen3/Eigen/Dense>
#include <random>
#include <string>

namespace SubsetSelection {

/*
Base matrix generator class
*/
template <typename scalar> class MatrixGenerator {
  protected:
    std::mt19937 gen;
    const std::pair<uint, uint> matrixSize;

  public:
    MatrixGenerator(uint m, uint n);
    MatrixGenerator(uint m, uint n, int seed);

    std::pair<uint, uint> getMatrixSize() const;
    virtual std::string getMatrixType() const;

    virtual Eigen::MatrixX<scalar> generateMatrix();
};

/*
Generator of a random matrix with orthonormal columns/rows
(depending on which of m and n is larger)
*/
template <typename scalar>
class OrthonormalVectorsMatrixGenerator : public MatrixGenerator<scalar> {
  public:
    OrthonormalVectorsMatrixGenerator(uint m, uint n);
    OrthonormalVectorsMatrixGenerator(uint m, uint n, int seed);

    std::string getMatrixType() const override;

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

    std::string getMatrixType() const override;

    Eigen::MatrixX<scalar> generateMatrix() override;
};

/*
Generator of a random matrix with the first singular value equal to 1
and other to eps (nearly rank-one matrix)
*/
template <typename scalar>
class NearRankOneMatrixGenerator : public SigmaMatrixGenerator<scalar> {
  private:
    scalar eps;
    Eigen::VectorX<scalar> getSigma(uint m, uint n, scalar eps) const;

  public:
    NearRankOneMatrixGenerator(uint m, uint n, scalar eps);
    NearRankOneMatrixGenerator(uint m, uint n, scalar eps, int seed);

    std::string getMatrixType() const override;
};

/*
Generator of a random matrix with the first min(m, n) - 1 singular
values equal to 1 and the remaining one equal to eps (matrix being near to
singular)
*/
template <typename scalar>
class NearSingularMatrixGenerator : public SigmaMatrixGenerator<scalar> {
  private:
    scalar eps;
    Eigen::VectorX<scalar> getSigma(uint m, uint n, scalar eps) const;

  public:
    NearSingularMatrixGenerator(uint m, uint n, scalar eps);
    NearSingularMatrixGenerator(uint m, uint n, scalar eps, int seed);

    std::string getMatrixType() const override;
};

/*
Generator of a matrix of right singular vectors of an edge-vertex incidence
matrix of a random fully-connected graph with m + 1 verticies and n edges.
*/
template <typename scalar>
class GraphIncidenceMatrixGenerator : public MatrixGenerator<scalar> {
  private:
    std::vector<std::pair<uint, uint>> randomEdgeList();

    bool checkConnectivity(const std::vector<std::pair<uint, uint>> &edge_list) const;

  protected:
    Eigen::MatrixX<scalar> incidenceMatrix();

  public:
    GraphIncidenceMatrixGenerator(uint m, uint n);
    GraphIncidenceMatrixGenerator(uint m, uint n, int seed);

    std::string getMatrixType() const override;

    Eigen::MatrixX<scalar> generateMatrix() override;
};

/*
Generator of a matrix of right singular vectors of an edge-vertex incidence
matrix of a random weighted fully-connected graph with m + 1 verticies and n
edges.
*/
template <typename scalar>
class WeightedGraphIncidenceMatrixGenerator
    : public GraphIncidenceMatrixGenerator<scalar> {
  public:
    WeightedGraphIncidenceMatrixGenerator(uint m, uint n);
    WeightedGraphIncidenceMatrixGenerator(uint m, uint n, int seed);

    std::string getMatrixType() const override;

    Eigen::MatrixX<scalar> generateMatrix() override;
};

/*
Generator of a matrix encountered when solving Smoluchowski equations. This
matrix usually presents problems for approximation algorithms.
*/
template <typename scalar>
class SmoluchowskiMatrixGenerator : public MatrixGenerator<scalar> {
  public:
    SmoluchowskiMatrixGenerator(uint m, uint n);
    SmoluchowskiMatrixGenerator(uint m, uint n, int seed);

    std::string getMatrixType() const override;

    Eigen::MatrixX<scalar> generateMatrix() override;
};

} // namespace SubsetSelection

#include "../src/matrix_generator.hpp"

#endif
