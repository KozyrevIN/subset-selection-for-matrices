#ifndef MAT_SUBSET_EXPERIMENTS_SNAPSHOT_SAVER_H
#define MAT_SUBSET_EXPERIMENTS_SNAPSHOT_SAVER_H

#include <algorithm>   // For std::fill
#include <cassert>     // For assert
#include <cstddef>     // For std::size_t
#include <filesystem>  // For std::filesystem::path, create_directories
#include <iomanip>     // For std::setw, std::setfill
#include <memory>      // For std::unique_ptr
#include <sstream>     // For std::ostringstream
#include <string>      // For std::string
#include <type_traits> // For std::conditional_t, std::is_same_v
#include <utility>     // For std::move
#include <vector>      // For std::vector

#include <Eigen/Core> // For Eigen::Index

#include <vtkAbstractArray.h>      // For vtkAbstractArray
#include <vtkDoubleArray.h>        // For vtkDoubleArray
#include <vtkFieldData.h>          // For vtkFieldData
#include <vtkFloatArray.h>         // For vtkFloatArray
#include <vtkImageData.h>          // For vtkImageData
#include <vtkNew.h>                // For vtkNew
#include <vtkPointData.h>          // For vtkPointData
#include <vtkSmartPointer.h>       // For vtkSmartPointer
#include <vtkXMLImageDataWriter.h> // For vtkXMLImageDataWriter, compressor

#include "TTCrossSolver/TensorTrain.h" // For TensorTrain

namespace MatSubset::Experiments {

/*!
 * @brief A regular d-dimensional grid on the box \f$ [a_1, b_1] \times \dots
 * \times [a_d, b_d] \f$, endpoints included: along dimension \f$ k \f$ with
 * \f$ n_k \f$ points, node \f$ i \f$ sits at \f$ a_k + i \, h_k \f$ with
 * \f$ h_k = (b_k - a_k) / (n_k - 1) \f$.
 * @tparam Scalar The underlying scalar type (e.g. `float`, `double`).
 *
 * Purely geometric: maps spatial multi-indices to coordinates and knows
 * nothing about tensors. The grid-index-to-tensor-index translation lives in
 * `FieldMapperBase` (identity for plain trains, bit folding for QTT, ...).
 */
template <typename Scalar> class Grid {
  public:
    /*!
     * @brief Constructs the grid from per-dimension point counts and box
     * corners.
     * @param sizes Points per dimension \f$ (n_1, \dots, n_d) \f$, each >= 1.
     * @param lower The box corner \f$ (a_1, \dots, a_d) \f$.
     * @param upper The box corner \f$ (b_1, \dots, b_d) \f$.
     */
    Grid(std::vector<Eigen::Index> sizes, std::vector<Scalar> lower,
         std::vector<Scalar> upper)
        : sizes(std::move(sizes)), lower(std::move(lower)),
          upper(std::move(upper)) {
        assert(this->sizes.size() == this->lower.size() &&
               this->sizes.size() == this->upper.size() &&
               "Grid: sizes, lower and upper must have the same dimension.");
        for (std::size_t k = 0; k < this->sizes.size(); ++k) {
            assert(this->sizes[k] >= 1 &&
                   "Grid: every dimension needs at least one point.");
        }
    }

    /*! @brief The spatial dimension d. */
    [[nodiscard]] std::size_t dim() const { return sizes.size(); }

    /*! @brief The number of points along dimension `k`. */
    [[nodiscard]] Eigen::Index size(std::size_t k) const {
        assert(k < sizes.size() && "Grid: dimension index out of range.");
        return sizes[k];
    }

    /*!
     * @brief The node spacing \f$ h_k = (b_k - a_k) / (n_k - 1) \f$ along
     * dimension `k`; 1 by convention for a single-point dimension (the only
     * node sits at \f$ a_k \f$ regardless).
     */
    [[nodiscard]] Scalar spacing(std::size_t k) const {
        assert(k < sizes.size() && "Grid: dimension index out of range.");
        return (sizes[k] > 1)
                   ? (upper[k] - lower[k]) / static_cast<Scalar>(sizes[k] - 1)
                   : Scalar(1);
    }

    /*! @brief The coordinate \f$ a_k + i \, h_k \f$ of node `i` along
     * dimension `k`. */
    [[nodiscard]] Scalar coordinate(std::size_t k, Eigen::Index i) const {
        assert(k < sizes.size() && "Grid: dimension index out of range.");
        assert(i >= 0 && i < sizes[k] && "Grid: node index out of range.");
        return lower[k] + static_cast<Scalar>(i) * spacing(k);
    }

    /*! @brief The spatial point of a grid multi-index. */
    [[nodiscard]] std::vector<Scalar>
    point(const std::vector<Eigen::Index> &idx) const {
        assert(idx.size() == sizes.size() &&
               "Grid: multi-index length must equal the dimension.");
        std::vector<Scalar> x(idx.size());
        for (std::size_t k = 0; k < idx.size(); ++k) {
            x[k] = coordinate(k, idx[k]);
        }
        return x;
    }

  private:
    std::vector<Eigen::Index> sizes;
    std::vector<Scalar> lower;
    std::vector<Scalar> upper;
};

/*!
 * @brief Base class for named scalar fields evaluated from a TT state at a
 * grid point.
 * @tparam Scalar The underlying scalar type (e.g. `float`, `double`).
 *
 * A mapper receives the *grid* multi-index and is responsible for translating
 * it into tensor indices: identity when the train's modes are the grid
 * dimensions, bit splitting for QTT, fixing a leading component mode for
 * stacked fields, or arbitrary combinations (e.g. an energy density built
 * from several components).
 */
template <typename Scalar> class FieldMapperBase {
  public:
    /*! @brief Stores the field name the snapshot writer labels the data with.
     */
    explicit FieldMapperBase(std::string name) : field_name(std::move(name)) {}

    virtual ~FieldMapperBase() = default;

    /*! @brief The field name (the VTK data array label). */
    [[nodiscard]] const std::string &name() const { return field_name; }

    /*!
     * @brief Evaluates the field from `state` at the grid node `grid_index`.
     */
    [[nodiscard]] virtual Scalar
    evaluate(const TensorTrain<Scalar> &state,
             const std::vector<Eigen::Index> &grid_index) const = 0;

  private:
    std::string field_name;
};

/*!
 * @brief The trivial field: grid multi-index equals tensor multi-index, the
 * field value is the train's entry there.
 */
template <typename Scalar>
class IdentityFieldMapper : public FieldMapperBase<Scalar> {
  public:
    using FieldMapperBase<Scalar>::FieldMapperBase;

    [[nodiscard]] Scalar
    evaluate(const TensorTrain<Scalar> &state,
             const std::vector<Eigen::Index> &grid_index) const override {
        return state(grid_index);
    }
};

/*!
 * @brief A decimated view of a fine tensor onto a coarser display grid:
 * display node \f$ g \f$ reads the tensor at
 * \f$ i_k = \text{offset}_k + \text{stride}_k \, g_k \f$.
 * @tparam Scalar The underlying scalar type (e.g. `float`, `double`).
 *
 * For visualizing a large tensor (say \f$ 1000^3 \f$) on a manageable grid
 * (say \f$ 200^3 \f$): `stride` is the per-axis decimation factor, `offset`
 * the first fine index sampled. The display grid must be small enough that
 * every mapped index stays in range — the mapper asserts on any out-of-bounds
 * node rather than clamping it.
 *
 * @note The display `Grid`'s spacing should be the fine field's spacing times
 * `stride` for coordinates to line up; that consistency is the caller's
 * responsibility, not enforced here.
 */
template <typename Scalar>
class SparseFieldMapper : public FieldMapperBase<Scalar> {
  public:
    /*!
     * @brief Constructs the mapper from per-axis stride and offset.
     * @param name The field name.
     * @param stride Per-axis decimation factor \f$ \text{stride}_k \ge 1 \f$;
     * its length is the tensor order.
     * @param offset Per-axis first fine index, same length as `stride`
     * (default all zeros).
     */
    SparseFieldMapper(std::string name, std::vector<Eigen::Index> stride,
                      std::vector<Eigen::Index> offset = {})
        : FieldMapperBase<Scalar>(std::move(name)), stride(std::move(stride)),
          offset(std::move(offset)) {
        if (this->offset.empty()) {
            this->offset.assign(this->stride.size(), Eigen::Index(0));
        }
        assert(this->stride.size() == this->offset.size() &&
               "SparseFieldMapper: stride and offset must have the same "
               "length.");
        for (std::size_t k = 0; k < this->stride.size(); ++k) {
            assert(this->stride[k] >= 1 &&
                   "SparseFieldMapper: stride must be at least 1.");
            assert(this->offset[k] >= 0 &&
                   "SparseFieldMapper: offset must be non-negative.");
        }
    }

    [[nodiscard]] Scalar
    evaluate(const TensorTrain<Scalar> &state,
             const std::vector<Eigen::Index> &grid_index) const override {
        const std::vector<Eigen::Index> sizes = state.modeSizes();
        assert(grid_index.size() == stride.size() &&
               sizes.size() == stride.size() &&
               "SparseFieldMapper: stride, grid index and tensor order must "
               "all match.");
        std::vector<Eigen::Index> tensor_index(grid_index.size());
        for (std::size_t k = 0; k < grid_index.size(); ++k) {
            tensor_index[k] = offset[k] + stride[k] * grid_index[k];
            assert(tensor_index[k] < sizes[k] &&
                   "SparseFieldMapper: display grid maps outside the tensor "
                   "bounds; shrink the grid or the stride/offset.");
        }
        return state(tensor_index);
    }

  private:
    std::vector<Eigen::Index> stride;
    std::vector<Eigen::Index> offset;
};

/*!
 * @brief Storage precision for the point-data arrays written to disk,
 * independent of the compute `Scalar`.
 */
enum class StoragePrecision {
    MatchScalar, //!< Store as `Scalar` (float -> float, double -> double).
    Float,       //!< Always store single precision (smaller files).
    Double       //!< Always store double precision.
};

/*!
 * @brief Writes snapshots of a TT state to VTK XML image files (`.vti`), one
 * scalar point-data array per stored mapper.
 * @tparam Scalar The underlying scalar type (e.g. `float`, `double`).
 *
 * Each `save` call evaluates every mapper at every grid node, assembles a
 * `vtkImageData` (a regular grid is image data in VTK terms) and writes
 * `<directory>/<basename>_<counter>.vti` with the snapshot time embedded as a
 * `TimeValue` field-data array (the ParaView convention, picked up as the
 * time axis), then bumps the counter. Grids of dimension < 3 are padded with
 * singleton axes; VTK caps the dimension at 3.
 *
 * Files are written appended-binary with ZLib compression, and the point-data
 * arrays are stored at the chosen `StoragePrecision` (default: match
 * `Scalar`). Dumping a `double` solve as `Float` roughly halves the payload
 * before compression at no cost to visualization; the `TimeValue` axis is
 * always kept double.
 *
 * @note This loops over the full grid, evaluating the train per point — fine
 * for output, but it is the one place the curse of dimensionality is paid.
 */
template <typename Scalar> class SnapshotSaver {
  public:
    /*!
     * @brief Constructs a saver from the grid, the fields to write, and the
     * output location.
     * @param grid The spatial grid the fields are sampled on (dim <= 3).
     * @param fields The field mappers, one scalar array per entry.
     * @param directory Output directory; created on save if missing.
     * @param basename File name prefix; files are `<basename>_NNNN.vti`.
     * @param precision On-disk precision of the point-data arrays; defaults to
     * matching `Scalar`. Use `Float` to shrink a `double` solve's output.
     */
    SnapshotSaver(Grid<Scalar> grid,
                  std::vector<std::unique_ptr<FieldMapperBase<Scalar>>> fields,
                  std::string directory, std::string basename = "snapshot",
                  StoragePrecision precision = StoragePrecision::MatchScalar)
        : grid(std::move(grid)), fields(std::move(fields)),
          directory(std::move(directory)), basename(std::move(basename)),
          precision(precision) {
        assert(this->grid.dim() >= 1 && this->grid.dim() <= 3 &&
               "SnapshotSaver: VTK image data supports 1 to 3 dimensions.");
        assert(!this->fields.empty() &&
               "SnapshotSaver: at least one field mapper is required.");
        for (const auto &field : this->fields) {
            assert(field != nullptr && "SnapshotSaver: null field mapper.");
        }
    }

    /*!
     * @brief Writes one snapshot of `state` at time `time` and increments the
     * file counter.
     * @return The path of the written file.
     */
    std::string save(const TensorTrain<Scalar> &state, Scalar time) {
        std::filesystem::create_directories(directory);

        vtkNew<vtkImageData> image;
        int dims[3] = {1, 1, 1};
        double origin[3] = {0.0, 0.0, 0.0};
        double spacing[3] = {1.0, 1.0, 1.0};
        for (std::size_t k = 0; k < grid.dim(); ++k) {
            dims[k] = static_cast<int>(grid.size(k));
            origin[k] = static_cast<double>(grid.coordinate(k, 0));
            spacing[k] = static_cast<double>(grid.spacing(k));
        }
        image->SetDimensions(dims);
        image->SetOrigin(origin);
        image->SetSpacing(spacing);

        // One point-data array per field, at the chosen storage precision.
        for (const auto &field : fields) {
            image->GetPointData()->AddArray(fillFieldArray(*field, state));
        }
        // Default coloring in ParaView: the first field.
        image->GetPointData()->SetActiveScalars(fields.front()->name().c_str());

        // The snapshot time, as the field-data array ParaView reads a time
        // axis from; always double, independent of the storage precision.
        vtkNew<vtkDoubleArray> time_value;
        time_value->SetName("TimeValue");
        time_value->SetNumberOfTuples(1);
        time_value->SetValue(0, static_cast<double>(time));
        image->GetFieldData()->AddArray(time_value);

        std::ostringstream file_name;
        file_name << basename << '_' << std::setw(4) << std::setfill('0')
                  << counter << ".vti";
        const std::filesystem::path path =
            std::filesystem::path(directory) / file_name.str();

        vtkNew<vtkXMLImageDataWriter> writer;
        writer->SetFileName(path.string().c_str());
        writer->SetInputData(image);
        // Appended raw binary + ZLib: the smallest widely-readable layout.
        writer->SetDataModeToAppended();
        writer->EncodeAppendedDataOff();
        writer->SetCompressorTypeToZLib();
        [[maybe_unused]] const int ok = writer->Write();
        assert(ok == 1 && "SnapshotSaver: vtkXMLImageDataWriter failed.");

        ++counter;
        return path.string();
    }

  private:
    Grid<Scalar> grid;
    std::vector<std::unique_ptr<FieldMapperBase<Scalar>>> fields;
    std::string directory;
    std::string basename;
    StoragePrecision precision;
    std::size_t counter = 0;

    /*!
     * @brief Evaluates one field over the whole grid into a VTK point-data
     * array of the configured storage precision.
     *
     * Filled in VTK point order (axis 0 fastest, then 1, then 2 — an odometer
     * over the grid multi-index). `MatchScalar` picks the array type from
     * `Scalar`; `Float`/`Double` override it, casting each value on store.
     */
    vtkSmartPointer<vtkAbstractArray>
    fillFieldArray(const FieldMapperBase<Scalar> &field,
                   const TensorTrain<Scalar> &state) const {
        const bool store_float = (precision == StoragePrecision::Float) ||
                                 (precision == StoragePrecision::MatchScalar &&
                                  std::is_same_v<Scalar, float>);
        if (store_float) {
            return fillTyped<vtkFloatArray>(field, state);
        }
        return fillTyped<vtkDoubleArray>(field, state);
    }

    /*! @brief `fillFieldArray` for a concrete VTK array type. */
    template <typename VtkArray>
    vtkSmartPointer<vtkAbstractArray>
    fillTyped(const FieldMapperBase<Scalar> &field,
              const TensorTrain<Scalar> &state) const {
        int dims[3] = {1, 1, 1};
        for (std::size_t k = 0; k < grid.dim(); ++k) {
            dims[k] = static_cast<int>(grid.size(k));
        }
        const auto total = static_cast<vtkIdType>(dims[0]) * dims[1] * dims[2];

        vtkNew<VtkArray> data;
        data->SetName(field.name().c_str());
        data->SetNumberOfComponents(1);
        data->SetNumberOfTuples(total);

        std::vector<Eigen::Index> idx(grid.dim(), 0);
        for (vtkIdType flat = 0; flat < total; ++flat) {
            data->SetValue(flat, field.evaluate(state, idx));
            for (std::size_t k = 0; k < idx.size(); ++k) {
                if (++idx[k] < grid.size(k)) {
                    break;
                }
                idx[k] = 0;
            }
        }
        return data;
    }
};

} // namespace MatSubset::Experiments

#endif // MAT_SUBSET_EXPERIMENTS_SNAPSHOT_SAVER_H
