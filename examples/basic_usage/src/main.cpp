#include <iostream>
#include <vector>

// Eigen is required, since MatSubset uses Eigen's matrices and algorithms.
// Eigen/Core includes Matrix and Array classes and basic operations with them.
#include <Eigen/Core>

// You can include all algorithms provided by MatSubset via
// #include <MatSubset/MatSubset.h>
// however it is recommended to include only neccesary ones to reduce compile
// time, for this example we stick with SpectralSelectionSelector.
#include <MatSubset/SpectralSelectionSelector.h>
// See documentation for full list of available algorithms and their properties.

int main() {
    // Our algorithms are template classes that depend on a scalar template
    // parameter. Template parameter must match those of matrix on which
    // algorithm will be applied. For now only float and double are supported
    // and tested, but some algorithms may work with complex numbers.
    using scalar = double;

    // To use algorithm you need to construct an object representing it.
    MatSubset::SpectralSelectionSelector<scalar> selector;

    // Every algorithm has a name, let us output it.
    std::cout << "Basic usage example of " << selector.getAlgorithmName()
              << " algorithm\n\n";

    // Now let us define a double valued matrix of size m times n. The matrix
    // must have more columns then rows (m <= n) and have full rank for
    // algorithms to work correctly.
    const int m = 2;
    const int n = 5;
    Eigen::MatrixX<scalar> matrix(m, n);
    matrix << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
    std::cout << "matrix =\n" << matrix << "\n\n";

    // Number of columns to select must be from m to n.
    const int k = 3;

    // Now we can apply the algorithm to select k columns from matrix.
    std::vector<uint> indices = selector.selectSubset(matrix, k);

    // Algorithm returns vector of indices of selected columns (0-based). The
    // order of indices it not specified.
    std::cout << "indices of selected columns = ";
    for (uint index : indices) {
        std::cout << index << ' ';
    }
    std::cout << "\n\n";

    // To acquire column submatrix you can use Eigen's slicing mechanism.
    Eigen::MatrixX<scalar> submatrix = matrix(Eigen::all, indices);
    std::cout << "selected submatrix =\n" << submatrix << "\n\n";

    return 0;
}