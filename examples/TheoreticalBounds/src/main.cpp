#include <cassert>
#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <MatSubset/SpectralSelectionSelector.h>
// For pinv_norm function which allows to compute pseudoinverse norm.
#include <MatSubset/Utils>

int main() {
    std::cout << "Theoretical bounds example\n\n";

    // Let us repeat several steps from basic_usage example
    using scalar = double;

    MatSubset::SpectralSelectionSelector<double> selector;

    const Eigen::Index m = 2;
    const Eigen::Index n = 5;
    const Eigen::Index k = 3;

    Eigen::MatrixX<scalar> matrix(m, n);
    // clang-format off
    matrix << 1, 2, 3, 4, 5,
              6, 7, 8, 9, 10;
    // clang-format on
    std::cout << "matrix =\n" << matrix << "\n\n";

    std::vector<Eigen::Index> indices = selector.selectSubset(matrix, k);
    Eigen::MatrixX<scalar> submatrix = matrix(Eigen::all, indices);
    std::cout << "selected submatrix =\n" << submatrix << "\n\n";

    // let us denote the initial m by n matrix by $X$, and selected submatrix of
    // k columns by $X_S$. Algorithms in MatSubset library provide theoretical
    // bounds on $\xi$ (spectral or Frobenius) norm of pseudoinverse of $X_S$ in
    // the following form
    //$\Vert X \Vert_\xi / \Vert X_S \Vert \geq bound_\xi(m, n, k)$
    // (Inverse to what is used in most artices on subset selection! This is
    // done to support algorithms that do not provide guarantees. For them, the
    // bound is simply 0). In this example we choose spectral norm.
    const MatSubset::Norm xi = MatSubset::Norm::Spectral;

    // You can acquire the bounds in the following way:
    scalar xi_norm_bound_1 = selector.bound<xi>(m, n, k);
    // Or this way:
    scalar xi_norm_bound_2 = selector.bound<xi>(matrix, k);

    // Bound depends only on matrix dimensions and k
    assert((xi_norm_bound_1 == xi_norm_bound_2) &&
           "Bound depends only on m, n, and k.");
    scalar xi_norm_bound = xi_norm_bound_1;

    // Let us calculate norm of pseudoinverse of original matrix and selected
    // submatrix
    scalar xi_norm_pinv_matrix =
        MatSubset::Utils::pinv_norm<scalar, xi>(matrix);
    scalar xi_norm_pinv_submatrix =
        MatSubset::Utils::pinv_norm<scalar, xi>(submatrix);

    // And check that theoretical bounds are satisfied
    if (xi_norm_bound == 0) {
        std::cout << "there are no guarantees on xi norm of pseudoinverse of "
                     "selected submatrix.\n\n";
    } else {
        std::cout << "xi norm of pseudoinverse of selected submatrix = "
                  << xi_norm_pinv_submatrix << ",\n"
                  << "while it must be <= "
                  << xi_norm_pinv_matrix / xi_norm_bound << "\n\n";

        if (xi_norm_pinv_submatrix <= xi_norm_pinv_matrix / xi_norm_bound) {
            std::cout << "theoretical bound is satisfied\n\n";
        } else {
            std::cout << "theoretical bound is NOT satisfied, consider "
                         "submitting an issue\n\n";
        }
    }

    return 0;
}