#include <eigen3/Eigen/Dense>
#include <string>

/*
Virtual base class for selecting k columns from m x n matrix (n >= k >= m) 
*/
template <typename scalar>
class SubsetSelector
{
private:

public:
    //constructor
    SubsetSelector(const std::string& algorithm_name);

    std::string algorithmName; 
    virtual std::vector<uint> selectSubset(const Eigen::MatrixX<scalar>& x, uint k);
};