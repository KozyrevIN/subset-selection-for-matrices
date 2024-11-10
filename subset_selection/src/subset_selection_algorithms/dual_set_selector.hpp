namespace SubsetSelection
{

template <typename scalar> 
DualSetSelector<scalar>::DualSetSelector(): SubsetSelector<scalar>("dual_set") {
    //do nothing
}

template <typename scalar> 
Eigen::ArrayX<scalar> DualSetSelector<scalar>::calculateL(const Eigen::MatrixX<scalar>& V, scalar delta_l,const Eigen::MatrixX<scalar>& A, scalar l) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<scalar>> decomposition(A);
    U = decomposition.eigenvectors();
    S = decomposition.eigenvalues().array();

    Eigen::MatrixX<scalar> M_1 = U * (S - (l + delta_l)).inverse().matrix().asDiagonal() * U.transpose();
    Eigen::MatrixX<scalar> M_2 = U * (S - (l + delta_l)).inverse().squared().matrix().asDiagonal() * U.transpose() / 
                                 ((S - (l + delta_l)).inverse().sum() - (S - l).inverse().sum());
    
    return (V.transpose() * (M_2 - M_1) * V).diagonal()
}

template <typename scalar> 
Eigen::ArrayX<scalar> DualSetSelector<scalar>::calculateU(scalar delta_u, const Eigen::ArrayX<scalar>& B, scalar u) {
    return ((u + delta_u) - B).inverse() + ((u + delta_u) - B).inverse().squared() /
           (((u + delta_u) - B).inverse().sum() - (u - B).inverse().sum());
}

template <typename scalar>
std::vector<uint> DualSetSelector<scalar>::selectSubset(const Eigen::MatrixX<scalar>& X, uint k) {
    uint m = X.rows();
    uint n = X.cols();

    Eigen::VectorX<scalar> s = Eigen::Zero(n);
    scalar delta_L = 1;
    scalar delts_U = (std::sqrt(n) - std::sqrt(k)) / (std::sqrt(k) - std::sqrt(m));

    

    return 0;
}

template <typename scalar>
scalar DualSetSelector<scalar>::bound(uint m, uint n, uint k, Norm norm) {
    return (std::pow(k, 0.5) - std::pow(m, 0.5)) / (std::pow(n, 0.5) - std::pow(k, 0.5));
}

}