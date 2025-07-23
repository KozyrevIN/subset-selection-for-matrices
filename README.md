# MatSubset: Subset Selection for Matrices

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight header-only C++ library implementing efficient approximation algorithms for subset selection for matrices, built on top of the Eigen linear algebra library.

## Installation

### Dependencies
For using library:
- [Eigen 3.3+](https://eigen.tuxfamily.org)

For running benchmarking code:
- [JsonCpp](https://github.com/open-source-parsers/jsoncpp)
- python 3 with numpy and matplotlib
  
For generating documentation:
- Doxygen
- Latex ()

### Using CMake Superbuild
```bash
cmake -S . -B build
cmake --build build
```

This will install the library to default location. Superbuild forwards user defined variables to all subprojects, so custom installation is also possible through superbuild. All other components (examples, tests, documentation, and, in future, benchmarks) are independent projects also reachable through superbuild by defining CMake variables (e. g. to additionally build documentation, pass ```-DBUILD_DOCS=ON```)

## Quick Start

Using implemenred algorithms is as easy as
```cpp
#include <MatSubset/...Selector.h> // Header with desired algorithm

using namespace MatSubset;

Eigen::MatrixXd X = ...;  // Your input matrix
Eigen::Index k = ...; // Number of columns to select
...Selector<double> selector;
std::vector<Eigen::Index> selected_indices = selector.selectSubset(X, k);
Eigen::MatrixXd X_S = X(Eigen::All, selected_indices); // Resulting submatrix
```
Explore complete examples in the [examples/](examples/) directory:
1. BasicUsage - Core API demonstration
2. TheoreticalBounds - Obtaining theoretical guarantees for algorithms

## Features

- Eigen-native API: Seamless integration with Eigen matrices
- Header-only design: Zero-compilation integration
- Modern algorithms: Implementation of state-of-the-art polynomial-time methods
- Extensible architecture: Easy to add custom selection strategies

## References

1. Avron & Boutsidis (2013) - [Faster subset selection for matrices](https://arxiv.org/abs/1307.0405)
2. Xie & Xu (2021) - [Fixed-block selection methods](https://doi.org/10.1137/1.9781611976472.15)
3. *Upcoming articles from our team* (2025)

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for full text.

> Permissions include commercial use, modification, and distribution. 
> Liability and warranty restrictions apply.
