namespace SubsetSelection
{

template <typename scalar> 
DualSetSelector<scalar>::DualSetSelector(): SubsetSelector<scalar>("dual_set") {
    //do nothing
}

template <typename scalar>
std::vector<uint> DualSetSelector<scalar>::selectSubset(const Eigen::MatrixX<scalar>& X, uint k) {
    std::vector<uint> cols(k);
    for (int i = 0; i < k; ++i) {
        cols[i] = i;
    }

    return cols;
}

template <typename scalar>
scalar DualSetSelector<scalar>::bound(uint m, uint n, uint k, Norm norm) {
    return 0;
}

}