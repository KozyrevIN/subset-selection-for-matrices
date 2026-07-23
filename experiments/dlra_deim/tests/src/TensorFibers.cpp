#include <doctest/doctest.h>

#include <memory>
#include <vector>

#include <Eigen/Core>

#include <TTCrossSolver/TensorFibers.h>

using MatSubset::Experiments::FiberIndices;
using MatSubset::Experiments::TensorFibers;

namespace {

// A 3-core skeleton, indexed by bond on both sides.
std::shared_ptr<const FiberIndices> makeSkeleton() {
    using Level = FiberIndices::Level;
    std::vector<Level> left(3), right(3);

    // left[k] is the bond-k left set; parents point into left[k-1], with -1
    // at left[0]. left[2] stays empty (no left set at the last bond).
    left[0] = Level({0}, {-1});
    left[1] = Level({0, 1}, {0, 0});

    // right[k] is the bond-k right set; parents point into right[k+1], with
    // the single-node root (the empty boundary index) at right[2].
    right[2] = Level({0}, {-1});
    right[1] = Level({0, 1}, {0, 0});
    right[0] = Level({0}, {0});

    return std::make_shared<const FiberIndices>(std::move(left),
                                                std::move(right));
}

// TensorFibers whose slab k has shape (leftFiberCount(k) * n_k) x
// rightFiberCount(k), filled with a constant.
template <typename Scalar>
TensorFibers<Scalar> makeFibers(std::shared_ptr<const FiberIndices> idx,
                                const std::vector<Eigen::Index> &modes,
                                Scalar fill) {
    std::vector<Eigen::MatrixX<Scalar>> slabs;
    for (std::size_t k = 0; k < idx->order(); ++k) {
        const auto rows =
            static_cast<Eigen::Index>(idx->leftFiberCount(k)) * modes[k];
        const auto cols = static_cast<Eigen::Index>(idx->rightFiberCount(k));
        slabs.push_back(Eigen::MatrixX<Scalar>::Constant(rows, cols, fill));
    }
    return TensorFibers<Scalar>(std::move(slabs), std::move(idx));
}

} // namespace

TEST_CASE("FiberIndices boundary fiber counts") {
    auto idx = makeSkeleton();
    CHECK(idx->order() == 3);
    // Core 0: left boundary -> 1 left fiber. Last core: right boundary -> 1.
    CHECK(idx->leftFiberCount(0) == 1);
    CHECK(idx->rightFiberCount(2) == 1);
    // Interior counts come from the neighbouring level sizes.
    CHECK(idx->leftFiberCount(1) == idx->leftLevel(0).size());
    CHECK(idx->rightFiberCount(1) == idx->rightLevel(1).size());
}

TEST_CASE("FiberIndices::Level exposes mode and parent") {
    auto idx = makeSkeleton();
    const auto &lvl = idx->leftLevel(1);
    CHECK(lvl.size() == 2);
    CHECK(lvl.mode(1) == 1);
    CHECK(lvl.parentOf(0) == 0);
    CHECK(idx->leftLevel(0).parentOf(0) == -1);  // left boundary
    CHECK(idx->rightLevel(2).parentOf(0) == -1); // right root at right[d-1]
    CHECK(idx->rightLevel(1).parentOf(0) == 0);  // extends the right root
}

TEST_CASE_TEMPLATE("TensorFibers operator+ and hadamardProduct are slab-wise",
                   Scalar, float, double) {
    auto idx = makeSkeleton();
    std::vector<Eigen::Index> modes{3, 4, 2};

    TensorFibers<Scalar> a = makeFibers<Scalar>(idx, modes, Scalar(2));
    TensorFibers<Scalar> b = makeFibers<Scalar>(idx, modes, Scalar(5));

    TensorFibers<Scalar> sum = a + b;
    TensorFibers<Scalar> prod = hadamardProduct(a, b);

    REQUIRE(sum.order() == a.order());
    REQUIRE(prod.order() == a.order());
    CHECK(sum.skeleton() == idx); // shares the skeleton

    for (std::size_t k = 0; k < a.order(); ++k) {
        CHECK(sum.core(k).leftUnfolding().rows() == a.core(k).leftUnfolding().rows());
        CHECK(sum.core(k).leftUnfolding().cols() == a.core(k).leftUnfolding().cols());
        CHECK((sum.core(k).leftUnfolding().array() == Scalar(7)).all());
        CHECK((prod.core(k).leftUnfolding().array() == Scalar(10)).all());
    }
}

TEST_CASE_TEMPLATE("TensorFibers operator* scales every slab", Scalar, float,
                   double) {
    auto idx = makeSkeleton();
    std::vector<Eigen::Index> modes{3, 4, 2};

    TensorFibers<Scalar> a = makeFibers<Scalar>(idx, modes, Scalar(2));
    TensorFibers<Scalar> scaled = Scalar(3) * a;

    REQUIRE(scaled.order() == a.order());
    CHECK(scaled.skeleton() == idx); // shares the skeleton

    for (std::size_t k = 0; k < a.order(); ++k) {
        CHECK(scaled.core(k).leftUnfolding().rows() == a.core(k).leftUnfolding().rows());
        CHECK(scaled.core(k).leftUnfolding().cols() == a.core(k).leftUnfolding().cols());
        CHECK((scaled.core(k).leftUnfolding().array() == Scalar(6)).all());
    }
}
