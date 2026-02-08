#include <doctest/doctest.h>

#include <Eigen/Eigenvalues>
#include <MatSubset/Utils.h>

// Helper function to compute eigenvalues using direct method
template <typename Scalar>
Eigen::VectorX<Scalar>
computeEigenvaluesDirect(const Eigen::VectorX<Scalar> &d,
                         const Eigen::VectorX<Scalar> &v) {
    Eigen::MatrixX<Scalar> M = v * v.transpose();
    M.diagonal() += d;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<Scalar>> solver(M);
    return solver.eigenvalues();
}

// Helper function to compare two vectors within tolerance
template <typename Scalar>
void checkEigenvaluesMatch(
    const Eigen::VectorX<Scalar> &computed,
    const Eigen::VectorX<Scalar> &expected,
    Scalar tol = 32 * std::numeric_limits<Scalar>::epsilon()) {
    REQUIRE(computed.size() == expected.size());
    for (Eigen::Index i = 0; i < computed.size(); ++i) {
        CHECK(std::abs(computed(i) - expected(i)) <
              tol * std::abs(expected(i)));
    }
}

TEST_CASE_TEMPLATE("SecularEquationSolver: basic case", Scalar, float, double) {
    // Simple case: distinct diagonal elements, all v components non-zero
    Eigen::VectorX<Scalar> d(3);
    d << 1.0, 2.0, 3.0;

    Eigen::VectorX<Scalar> v(3);
    v << 0.5, 0.3, 0.2;

    MatSubset::Utils::SecularEquationSolver<Scalar> solver;
    Eigen::VectorX<Scalar> computed = solver.solve(d, v);
    Eigen::VectorX<Scalar> expected = computeEigenvaluesDirect(d, v);

    checkEigenvaluesMatch(computed, expected);
}

TEST_CASE_TEMPLATE("SecularEquationSolver: deflation type 1 - zero v component",
                   Scalar, float, double) {
    // One v component is zero -> that eigenvalue stays unchanged
    Eigen::VectorX<Scalar> d(3);
    d << 1.0, 2.0, 3.0;

    Eigen::VectorX<Scalar> v(3);
    v << 0.5, 0.0, 0.2; // Middle component is zero

    MatSubset::Utils::SecularEquationSolver<Scalar> solver;
    Eigen::VectorX<Scalar> computed = solver.solve(d, v);
    Eigen::VectorX<Scalar> expected = computeEigenvaluesDirect(d, v);

    checkEigenvaluesMatch(computed, expected);
}

TEST_CASE_TEMPLATE("SecularEquationSolver: deflation type 1 - multiple zero v",
                   Scalar, float, double) {
    // Multiple v components are zero
    Eigen::VectorX<Scalar> d(4);
    d << 1.0, 2.0, 3.0, 4.0;

    Eigen::VectorX<Scalar> v(4);
    v << 0.0, 0.5, 0.0, 0.3; // First and third are zero

    MatSubset::Utils::SecularEquationSolver<Scalar> solver;
    Eigen::VectorX<Scalar> computed = solver.solve(d, v);
    Eigen::VectorX<Scalar> expected = computeEigenvaluesDirect(d, v);

    checkEigenvaluesMatch(computed, expected);
}

TEST_CASE_TEMPLATE(
    "SecularEquationSolver: deflation type 2 - repeated eigenvalues", Scalar,
    float, double) {
    // Two diagonal elements are equal
    Eigen::VectorX<Scalar> d(4);
    d << 1.0, 2.0, 2.0, 3.0; // Two equal elements

    Eigen::VectorX<Scalar> v(4);
    v << 0.3, 0.4, 0.5, 0.2;

    MatSubset::Utils::SecularEquationSolver<Scalar> solver;
    Eigen::VectorX<Scalar> computed = solver.solve(d, v);
    Eigen::VectorX<Scalar> expected = computeEigenvaluesDirect(d, v);

    checkEigenvaluesMatch(computed, expected);
}

TEST_CASE_TEMPLATE(
    "SecularEquationSolver: deflation type 2 - triple eigenvalue", Scalar,
    float, double) {
    // Three diagonal elements are equal
    Eigen::VectorX<Scalar> d(5);
    d << 1.0, 2.0, 2.0, 2.0, 3.0; // Three equal elements

    Eigen::VectorX<Scalar> v(5);
    v << 0.1, 0.3, 0.4, 0.5, 0.2;

    MatSubset::Utils::SecularEquationSolver<Scalar> solver;
    Eigen::VectorX<Scalar> computed = solver.solve(d, v);
    Eigen::VectorX<Scalar> expected = computeEigenvaluesDirect(d, v);

    checkEigenvaluesMatch(computed, expected);
}

TEST_CASE_TEMPLATE("SecularEquationSolver: combined deflation types", Scalar,
                   float, double) {
    // Both zero v and repeated eigenvalues
    Eigen::VectorX<Scalar> d(5);
    d << 1.0, 2.0, 2.0, 3.0, 4.0;

    Eigen::VectorX<Scalar> v(5);
    v << 0.3, 0.4, 0.0, 0.5, 0.0; // Zero in repeated cluster and elsewhere

    MatSubset::Utils::SecularEquationSolver<Scalar> solver;
    Eigen::VectorX<Scalar> computed = solver.solve(d, v);
    Eigen::VectorX<Scalar> expected = computeEigenvaluesDirect(d, v);

    checkEigenvaluesMatch(computed, expected);
}

TEST_CASE_TEMPLATE("SecularEquationSolver: all zeros in repeated cluster",
                   Scalar, float, double) {
    // All v components in a repeated cluster are zero
    Eigen::VectorX<Scalar> d(4);
    d << 1.0, 2.0, 2.0, 3.0;

    Eigen::VectorX<Scalar> v(4);
    v << 0.5, 0.0, 0.0, 0.3; // Both components in the cluster are zero

    MatSubset::Utils::SecularEquationSolver<Scalar> solver;
    Eigen::VectorX<Scalar> computed = solver.solve(d, v);
    Eigen::VectorX<Scalar> expected = computeEigenvaluesDirect(d, v);

    checkEigenvaluesMatch(computed, expected);
}

TEST_CASE_TEMPLATE("SecularEquationSolver: zero diagonal element", Scalar,
                   float, double) {
    // Diagonal contains zero
    Eigen::VectorX<Scalar> d(3);
    d << 0.0, 1.0, 2.0;

    Eigen::VectorX<Scalar> v(3);
    v << 0.5, 0.3, 0.2;

    MatSubset::Utils::SecularEquationSolver<Scalar> solver;
    Eigen::VectorX<Scalar> computed = solver.solve(d, v);
    Eigen::VectorX<Scalar> expected = computeEigenvaluesDirect(d, v);

    checkEigenvaluesMatch(computed, expected);
}

TEST_CASE_TEMPLATE("SecularEquationSolver: large vector", Scalar, float,
                   double) {
    // Larger test case
    const Eigen::Index n = 20;
    Eigen::VectorX<Scalar> d(n);
    Eigen::VectorX<Scalar> v(n);

    for (Eigen::Index i = 0; i < n; ++i) {
        d(i) = static_cast<Scalar>(i + 1);
        v(i) = static_cast<Scalar>(1.0 / (i + 1));
    }

    MatSubset::Utils::SecularEquationSolver<Scalar> solver;
    Eigen::VectorX<Scalar> computed = solver.solve(d, v);
    Eigen::VectorX<Scalar> expected = computeEigenvaluesDirect(d, v);

    checkEigenvaluesMatch(computed, expected);
}

TEST_CASE_TEMPLATE("SecularEquationSolver: single element", Scalar, float,
                   double) {
    // Edge case: single element
    Eigen::VectorX<Scalar> d(1);
    d << 2.0;

    Eigen::VectorX<Scalar> v(1);
    v << 0.5;

    MatSubset::Utils::SecularEquationSolver<Scalar> solver;
    Eigen::VectorX<Scalar> computed = solver.solve(d, v);
    Eigen::VectorX<Scalar> expected = computeEigenvaluesDirect(d, v);

    checkEigenvaluesMatch(computed, expected);
}

TEST_CASE_TEMPLATE("SecularEquationSolver: all equal diagonals", Scalar, float,
                   double) {
    // All diagonal elements are equal
    Eigen::VectorX<Scalar> d(4);
    d << 2.0, 2.0, 2.0, 2.0;

    Eigen::VectorX<Scalar> v(4);
    v << 0.5, 0.3, 0.4, 0.2;

    MatSubset::Utils::SecularEquationSolver<Scalar> solver;
    Eigen::VectorX<Scalar> computed = solver.solve(d, v);
    Eigen::VectorX<Scalar> expected = computeEigenvaluesDirect(d, v);

    checkEigenvaluesMatch(computed, expected);
}
