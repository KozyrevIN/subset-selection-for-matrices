#ifndef MAT_SUBSET_EXPERIMENTS_FINITE_DIFFERENCE_H
#define MAT_SUBSET_EXPERIMENTS_FINITE_DIFFERENCE_H

#include <cassert> // For assert
#include <cmath>   // For std::abs
#include <cstddef> // For std::size_t
#include <vector>  // For std::vector

namespace MatSubset::Experiments {

/*!
 * @brief The symmetric central finite-difference weights of the second
 * derivative at accuracy order \f$ 2m \f$, offsets \f$ 0 \dots m \f$ (unit
 * spacing).
 * @tparam Scalar The underlying scalar type (e.g. `float`, `double`).
 * @param order The (even) accuracy order \f$ 2m \ge 2 \f$.
 * @return `w` of length \f$ m + 1 \f$: `w[0]` is the center weight and `w[j]`
 * (for \f$ j \ge 1 \f$) the weight shared by both \f$ \pm j \f$ neighbours, so
 * \f$ f''(x_i) \approx h^{-2} \bigl( w_0 f_i + \sum_{j\ge1} w_j (f_{i-j} +
 * f_{i+j}) \bigr) \f$.
 *
 * The closed form of the \f$ (2m+1) \f$-point central stencil:
 * \f$ w_j = \dfrac{2\,(-1)^{j+1} (m!)^2}{(m-j)!\,(m+j)!\,j^2} \f$ for
 * \f$ j \ge 1 \f$ and \f$ w_0 = -2 \sum_{j\ge1} w_j \f$. Order 2 is
 * \f$ (1, -2, 1) \f$, order 4 is \f$ (-1, 16, -30, 16, -1)/12 \f$, order 6 is
 * \f$ (2, -27, 270, -490, 270, -27, 2)/180 \f$. The weights already include the
 * normalizing denominator, so only a division by \f$ h^2 \f$ remains.
 *
 * Higher orders cut the stencil's numerical dispersion, which the wave
 * operator otherwise accumulates over time — so a coarse and a refined grid
 * stay in phase far longer, the whole point of raising the order here.
 */
template <typename Scalar>
[[nodiscard]] std::vector<Scalar> centralSecondDerivativeWeights(int order) {
    assert(order >= 2 && order % 2 == 0 &&
           "centralSecondDerivativeWeights: order must be even and >= 2.");
    const int m = order / 2;

    // w_j = 2 (-1)^(j+1) (m!)^2 / ((m-j)! (m+j)! j^2), built as a running
    // product to keep the factorials exact in Scalar for the orders in use.
    std::vector<Scalar> w(static_cast<std::size_t>(m) + 1, Scalar(0));
    // ratio_j = (m!)^2 / ((m-j)! (m+j)!) = prod_{t=1..j} (m - t + 1) / (m + t).
    Scalar ratio = Scalar(1);
    Scalar sum_off = Scalar(0);
    for (int j = 1; j <= m; ++j) {
        ratio *= static_cast<Scalar>(m - j + 1) / static_cast<Scalar>(m + j);
        const Scalar sign = (j % 2 == 1) ? Scalar(1) : Scalar(-1);
        w[static_cast<std::size_t>(j)] =
            Scalar(2) * sign * ratio / static_cast<Scalar>(j * j);
        sum_off += w[static_cast<std::size_t>(j)];
    }
    w[0] = Scalar(-2) * sum_off;
    return w;
}

/*!
 * @brief The per-axis spectral factor \f$ S \f$ of the order-\f$ 2m \f$
 * second-derivative stencil: the largest \f$ |\lambda| h^2 \f$ of the 1D
 * operator, equal to the row absolute sum \f$ |w_0| + 2 \sum_{j\ge1} |w_j| \f$.
 * @tparam Scalar The underlying scalar type (e.g. `float`, `double`).
 * @param order The (even) accuracy order.
 * @return \f$ S \f$ (4 at order 2, 16/3 at order 4, ...); used to set the CFL
 * time step, since the \f$ d \f$-dimensional Laplacian's spectral radius is
 * \f$ d\,S / h^2 \f$.
 */
template <typename Scalar>
[[nodiscard]] Scalar secondDerivativeSpectralFactor(int order) {
    const std::vector<Scalar> w = centralSecondDerivativeWeights<Scalar>(order);
    Scalar s = std::abs(w[0]);
    for (std::size_t j = 1; j < w.size(); ++j) {
        s += Scalar(2) * std::abs(w[j]);
    }
    return s;
}

} // namespace MatSubset::Experiments

#endif // MAT_SUBSET_EXPERIMENTS_FINITE_DIFFERENCE_H
