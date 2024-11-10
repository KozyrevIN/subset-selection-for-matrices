#ifndef MATRIX_UTILITIES_H
#define MATRIX_UTILITIES_H

#include <eigen3/Eigen/Dense>

#include "enums.h"

namespace SubsetSelection {

template <typename scalar, Norm norm>
scalar pinv_norm(Eigen::MatrixX<scalar> X);

}

#include "../src/matrix_utilities.hpp"

#endif
