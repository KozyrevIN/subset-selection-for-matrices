#include <doctest/doctest.h>

#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include <vtkDataArray.h>
#include <vtkFieldData.h>
#include <vtkImageData.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkXMLImageDataReader.h>

#include <TTCrossSolver/SnapshotSaver.h>
#include <TTCrossSolver/TensorTrain.h>
#include <TTCrossSolver/TensorTrainCore.h>

using MatSubset::Experiments::FieldMapperBase;
using MatSubset::Experiments::Grid;
using MatSubset::Experiments::IdentityFieldMapper;
using MatSubset::Experiments::SnapshotSaver;
using MatSubset::Experiments::SparseFieldMapper;
using MatSubset::Experiments::StoragePrecision;
using MatSubset::Experiments::TensorTrain;
using MatSubset::Experiments::TensorTrainCore;

namespace {

// Deterministic, well-conditioned left unfolding (r0 * n) x r1.
template <typename Scalar>
Eigen::MatrixX<Scalar> makeUnfolding(Eigen::Index r0, Eigen::Index n,
                                     Eigen::Index r1, Eigen::Index salt) {
    Eigen::MatrixX<Scalar> A(r0 * n, r1);
    for (Eigen::Index i = 0; i < A.rows(); ++i) {
        for (Eigen::Index j = 0; j < A.cols(); ++j) {
            A(i, j) = static_cast<Scalar>(1 + (i * 7 + j * 3 + salt * 5) % 11);
        }
    }
    return A;
}

// Builds a 2-core train with mode sizes (n0, n1) and bond rank r.
template <typename Scalar>
TensorTrain<Scalar> makeTrain2(Eigen::Index n0, Eigen::Index n1,
                               Eigen::Index r) {
    std::vector<TensorTrainCore<Scalar>> cores;
    cores.emplace_back(makeUnfolding<Scalar>(1, n0, r, 0), n0);
    cores.emplace_back(makeUnfolding<Scalar>(r, n1, 1, 1), n1);
    return TensorTrain<Scalar>(std::move(cores));
}

template <typename Scalar> Scalar checkTol() {
    return std::is_same_v<Scalar, float> ? Scalar(1e-3) : Scalar(1e-9);
}

// A non-trivial mapping: twice the train's entry at the grid index.
template <typename Scalar>
class DoubledFieldMapper : public FieldMapperBase<Scalar> {
  public:
    using FieldMapperBase<Scalar>::FieldMapperBase;

    [[nodiscard]] Scalar
    evaluate(const TensorTrain<Scalar> &state,
             const std::vector<Eigen::Index> &grid_index) const override {
        return Scalar(2) * state(grid_index);
    }
};

} // namespace

TEST_CASE_TEMPLATE("Grid maps multi-indices to coordinates", Scalar, float,
                   double) {
    Grid<Scalar> grid({3, 5}, {Scalar(0), Scalar(-1)}, {Scalar(1), Scalar(1)});

    CHECK(grid.dim() == 2);
    CHECK(grid.size(0) == 3);
    CHECK(grid.size(1) == 5);

    // Endpoints included: h = (b - a) / (n - 1).
    CHECK(static_cast<double>(grid.spacing(0)) == doctest::Approx(0.5));
    CHECK(static_cast<double>(grid.spacing(1)) == doctest::Approx(0.5));
    CHECK(static_cast<double>(grid.coordinate(0, 0)) == doctest::Approx(0.0));
    CHECK(static_cast<double>(grid.coordinate(0, 2)) == doctest::Approx(1.0));
    CHECK(static_cast<double>(grid.coordinate(1, 0)) == doctest::Approx(-1.0));
    CHECK(static_cast<double>(grid.coordinate(1, 4)) == doctest::Approx(1.0));

    auto x = grid.point({1, 2});
    REQUIRE(x.size() == 2);
    CHECK(static_cast<double>(x[0]) == doctest::Approx(0.5));
    CHECK(static_cast<double>(x[1]) == doctest::Approx(0.0));

    // A single-point dimension sits at its lower bound.
    Grid<Scalar> flat({1}, {Scalar(3)}, {Scalar(7)});
    CHECK(static_cast<double>(flat.coordinate(0, 0)) == doctest::Approx(3.0));
}

TEST_CASE_TEMPLATE("IdentityFieldMapper evaluates the train and keeps its "
                   "name",
                   Scalar, float, double) {
    auto tt = makeTrain2<Scalar>(3, 4, 2);
    IdentityFieldMapper<Scalar> mapper("pressure");

    CHECK(mapper.name() == "pressure");
    const std::vector<Eigen::Index> idx{2, 3};
    CHECK(std::abs(mapper.evaluate(tt, idx) - tt(idx)) < checkTol<Scalar>());
}

TEST_CASE_TEMPLATE("SnapshotSaver writes VTK image files the reader round-"
                   "trips",
                   Scalar, float, double) {
    namespace fs = std::filesystem;

    const Eigen::Index n0 = 3, n1 = 2;
    auto tt = makeTrain2<Scalar>(n0, n1, 2);

    const std::string dir =
        (fs::temp_directory_path() / "ttcross_snapshot_test").string();
    fs::remove_all(dir);

    Grid<Scalar> grid({n0, n1}, {Scalar(0), Scalar(-1)},
                      {Scalar(1), Scalar(1)});
    std::vector<std::unique_ptr<FieldMapperBase<Scalar>>> fields;
    fields.push_back(std::make_unique<IdentityFieldMapper<Scalar>>("f"));
    fields.push_back(std::make_unique<DoubledFieldMapper<Scalar>>("g"));
    SnapshotSaver<Scalar> saver(std::move(grid), std::move(fields), dir,
                                "state");

    const Scalar t0 = Scalar(0.25);
    const std::string path0 = saver.save(tt, t0);
    const std::string path1 = saver.save(tt, Scalar(0.5));

    // The counter names the files in order.
    CHECK(path0.ends_with("state_0000.vti"));
    CHECK(path1.ends_with("state_0001.vti"));
    REQUIRE(fs::exists(path0));
    REQUIRE(fs::exists(path1));

    // Read the first snapshot back.
    vtkNew<vtkXMLImageDataReader> reader;
    reader->SetFileName(path0.c_str());
    reader->Update();
    vtkImageData *image = reader->GetOutput();
    REQUIRE(image != nullptr);

    // Geometry: the 2D grid padded to a singleton third axis.
    int dims[3];
    image->GetDimensions(dims);
    CHECK(dims[0] == n0);
    CHECK(dims[1] == n1);
    CHECK(dims[2] == 1);
    CHECK(image->GetOrigin()[0] == doctest::Approx(0.0));
    CHECK(image->GetOrigin()[1] == doctest::Approx(-1.0));
    CHECK(image->GetSpacing()[0] == doctest::Approx(0.5));
    CHECK(image->GetSpacing()[1] == doctest::Approx(2.0));

    // The snapshot time rides along as field data.
    vtkDataArray *time_value = image->GetFieldData()->GetArray("TimeValue");
    REQUIRE(time_value != nullptr);
    CHECK(time_value->GetTuple1(0) == doctest::Approx(static_cast<double>(t0)));

    // Point data: both fields, in VTK point order (axis 0 fastest), matching
    // the train's entries at the grid multi-indices.
    vtkDataArray *f = image->GetPointData()->GetArray("f");
    vtkDataArray *g = image->GetPointData()->GetArray("g");
    REQUIRE(f != nullptr);
    REQUIRE(g != nullptr);
    REQUIRE(f->GetNumberOfTuples() == n0 * n1);

    const double tol = static_cast<double>(checkTol<Scalar>());
    for (Eigen::Index i1 = 0; i1 < n1; ++i1) {
        for (Eigen::Index i0 = 0; i0 < n0; ++i0) {
            const auto flat = static_cast<vtkIdType>(i0 + n0 * i1);
            const double value =
                static_cast<double>(tt(std::vector<Eigen::Index>{i0, i1}));
            CHECK(f->GetTuple1(flat) == doctest::Approx(value).epsilon(tol));
            CHECK(g->GetTuple1(flat) ==
                  doctest::Approx(2 * value).epsilon(tol));
        }
    }

    fs::remove_all(dir);
}

TEST_CASE_TEMPLATE("SparseFieldMapper decimates the tensor by stride/offset",
                   Scalar, float, double) {
    // A fine 6x4 tensor viewed on a coarser display grid: stride (2, 3),
    // offset (1, 0) -> display node (g0, g1) reads tensor (1 + 2*g0, 3*g1).
    auto tt = makeTrain2<Scalar>(6, 4, 2);
    SparseFieldMapper<Scalar> mapper("u", {2, 3}, {1, 0});

    CHECK(mapper.name() == "u");
    // Display grid 3x2 stays in bounds: max tensor index (5, 3).
    for (Eigen::Index g0 = 0; g0 < 3; ++g0) {
        for (Eigen::Index g1 = 0; g1 < 2; ++g1) {
            const std::vector<Eigen::Index> expect{1 + 2 * g0, 3 * g1};
            CHECK(std::abs(mapper.evaluate(tt, {g0, g1}) - tt(expect)) <
                  checkTol<Scalar>());
        }
    }

    // Default offset is all zeros: pure decimation.
    SparseFieldMapper<Scalar> plain("u", {2, 2});
    CHECK(std::abs(plain.evaluate(tt, {2, 1}) -
                   tt(std::vector<Eigen::Index>{4, 2})) < checkTol<Scalar>());
}

TEST_CASE("SnapshotSaver Float precision shrinks a double solve on disk") {
    namespace fs = std::filesystem;

    const Eigen::Index n0 = 6, n1 = 6;
    auto tt = makeTrain2<double>(n0, n1, 3);

    const std::string dir =
        (fs::temp_directory_path() / "ttcross_precision_test").string();
    fs::remove_all(dir);
    Grid<double> grid({n0, n1}, {0.0, 0.0}, {1.0, 1.0});

    auto make_saver = [&](StoragePrecision precision, const std::string &base) {
        std::vector<std::unique_ptr<FieldMapperBase<double>>> fields;
        fields.push_back(std::make_unique<IdentityFieldMapper<double>>("f"));
        return SnapshotSaver<double>(grid, std::move(fields), dir, base,
                                     precision);
    };

    auto saver_double = make_saver(StoragePrecision::Double, "dbl");
    auto saver_float = make_saver(StoragePrecision::Float, "flt");
    const std::string dbl_path = saver_double.save(tt, 0.0);
    const std::string flt_path = saver_float.save(tt, 0.0);

    // Same field, half-width storage: the float file is smaller.
    CHECK(fs::file_size(flt_path) < fs::file_size(dbl_path));

    // Float storage still round-trips to visualization tolerance.
    vtkNew<vtkXMLImageDataReader> reader;
    reader->SetFileName(flt_path.c_str());
    reader->Update();
    vtkDataArray *f = reader->GetOutput()->GetPointData()->GetArray("f");
    REQUIRE(f != nullptr);
    for (Eigen::Index i1 = 0; i1 < n1; ++i1) {
        for (Eigen::Index i0 = 0; i0 < n0; ++i0) {
            const auto flat = static_cast<vtkIdType>(i0 + n0 * i1);
            const double value = tt(std::vector<Eigen::Index>{i0, i1});
            CHECK(f->GetTuple1(flat) == doctest::Approx(value).epsilon(1e-5));
        }
    }

    fs::remove_all(dir);
}
