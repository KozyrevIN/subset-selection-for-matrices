#include <eigen3/Eigen/Dense>
#include <string>

/*
Virtual base class for selecting k columns from m x n matrix (n >= k >= m) 
*/
template <typename scalar>
class SubsetSelector
{
public:
    const std::string algorithmName; 

    SubsetSelector(const std::string& algorithm_name);

    virtual std::vector<uint> selectSubset(const Eigen::MatrixX<scalar>& X, uint k);

    scalar frobeniusBound(const Eigen::MatrixX<scalar>& X, uint k);
    virtual scalar frobeniusBound(uint m, uint n, uint k);

    scalar l2Bound(const Eigen::MatrixX<scalar>& X, uint k);
    virtual scalar l2Bound(uint m, uint n, uint k);
};