#include <chrono>
#include <complex>

#include "../include/tester.h"
#include "../include/matrix_utilities.h"

template <typename scalar>
Tester<scalar>::Tester() {
    //do nothing
}

template <typename scalar>
std::string Tester<scalar>::testAlgorithmOnMatrix(const Eigen::MatrixX<scalar>& A, SubsetSelector<scalar>* algorithm, uint k) {
    std::string results;
    double time = 0; double volume_reduction = 0;
    for (uint i = 0; i < 100; ++i) {
        auto t1 = std::chrono::high_resolution_clock::now();
        auto subset = algorithm -> selectSubset(A, k);
        auto t2 = std::chrono::high_resolution_clock::now();
        double pinv_norm = pinv_frobenius_norm<scalar>(A(Eigen::all, subset));
        double pinv_norm_0 = pinv_frobenius_norm<scalar>(A);

        volume_reduction += pinv_norm_0 / pinv_norm;
        time += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    }
    time *= 0.01;
    volume_reduction *= 0.01;
    results = algorithm -> algorithmName + ":\n"
              "    volume reduction = " + std::to_string(volume_reduction) + "\n"
              "    time = " + std::to_string(time) + " ms \n";

    return results;
}

template <typename scalar>
std::string Tester<scalar>::testAlgorithmOnMatrix(MatrixGenerator<scalar>* mat_gen, SubsetSelector<scalar>* algorithm, uint k) {
    auto A = mat_gen -> generateMatrix();

    return testAlgorithmOnMatrix(A, algorithm, k);
}

template <typename scalar>
std::string Tester<scalar>::testAlgorithmsOnMatrix(const Eigen::MatrixX<scalar>& A, std::vector<SubsetSelector<scalar>*> algorithms, uint k) {
    std::string results;
    for (auto algorithm : algorithms) {
        results += testAlgorithmOnMatrix(A, algorithm, k);
    }

    return results;
}

template class Tester<float>;
template class Tester<double>;
template class Tester<std::complex<float>>;
template class Tester<std::complex<double>>;