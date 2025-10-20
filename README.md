# MatSubset: Subset Selection for Matrices

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight header-only C++ library implementing efficient approximation algorithms for subset selection problem for matrices, built on top of the Eigen linear algebra library.

## Installation

### Dependencies

**Required (library and examples):**
- [Eigen 3.3+](https://eigen.tuxfamily.org)

**Optional (tests, enabled with `-DBUILD_TESTS=ON`):**
- [doctest 2.4.12](https://github.com/doctest/doctest) (fetched automatically)

**Optional (documentation, enabled with `-DBUILD_DOCS=ON`):**
- [Doxygen](https://doxygen.nl/)
- [m.css](https://mcss.mosra.cz/) (fetched automatically)
- Python 3
- LaTeX with fonts (texlive-base, texlive-latex-extra, texlive-fonts-extra, texlive-fonts-recommended)

**Optional (benchmarks, enabled with `-DBUILD_BENCH=ON`):**
- [nlohmann/json 3.12.0](https://github.com/nlohmann/json) (fetched automatically)
- [OpenMP](https://www.openmp.org/)
- Python 3 with NumPy, pandas, Matplotlib (for plotter.py)
- LaTeX (for plot text rendering)

### Using CMake Superbuild
```bash
cmake -S . -B build
cmake --build build
```

This will install the library to default location. Superbuild forwards user defined variables to all subprojects, so custom installation is also possible through superbuild. All other components (examples, tests, documentation, and, in future, benchmarks) are independent projects also reachable through superbuild by defining CMake variables (e.g., to additionally build documentation, pass ```-DBUILD_DOCS=ON```)

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

1. Avron & Boutsidis (2013) - [Faster subset selection for matrices and applications](https://arxiv.org/abs/1307.0405)
2. Kozyrev & Osinsky (2025) - [Subset selection for matrices in spectral norm](https://arxiv.org/abs/2507.20435)
3. Xie & Xu (2021) - [Subset selection for matrices with fixed blocks](https://doi.org/10.1137/1.9781611976472.15)


## License

Distributed under the MIT License. See [LICENSE](LICENSE) for full text.

> Permissions include commercial use, modification, and distribution. 
> Liability and warranty restrictions apply.
