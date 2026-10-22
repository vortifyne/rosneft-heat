#include "discretization/nonmatching_column_hfv_system.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

using LocalMatrix = Eigen::MatrixXd;

void require_solution_size(const Vector& solution, const Vector::Index expected) {
    if (solution.size() != expected) {
        throw std::invalid_argument("HFV solution has an unexpected size");
    }
}

} // namespace

NonmatchingColumnHfvSystem::NonmatchingColumnHfvSystem(
    const NonmatchingColumnGrid& grid, const std::span<const double> thermal_conductivity,
    const std::span<const double> heat_production, NonmatchingColumnBoundaryData boundary_data,
    PropertyUpdater property_updater)
    : grid_(&grid), thermal_conductivity_(thermal_conductivity.begin(), thermal_conductivity.end()),
      heat_production_(heat_production.begin(), heat_production.end()),
      boundary_data_(std::move(boundary_data)), property_updater_(std::move(property_updater)),
      cell_faces_(grid.cells.size()) {
    if (thermal_conductivity_.size() != grid.cells.size() ||
        heat_production_.size() != grid.cells.size() || !boundary_data_.surface_temperature ||
        !boundary_data_.basal_inward_flux) {
        throw std::invalid_argument("HFV properties and boundary data do not match the grid");
    }
    for (const double value : thermal_conductivity_) {
        if (!std::isfinite(value) || !(value > 0.0)) {
            throw std::invalid_argument("HFV conductivity must be positive and finite");
        }
    }

    const auto append_face = [this](const Point2D center, const double length, const FaceKind kind,
                                    const std::initializer_list<CellFace> incidences) {
        if (!std::isfinite(center.x) || !std::isfinite(center.z) || !std::isfinite(length) ||
            !(length > 0.0)) {
            throw std::invalid_argument("HFV face geometry must be finite and non-degenerate");
        }
        const std::size_t face_index = faces_.size();
        Face face{
            .center = center, .length = length, .kind = kind, .unknown = std::nullopt, .cells = {}};
        for (const CellFace& incidence : incidences) {
            if (incidence.face >= grid_->cells.size()) {
                throw std::invalid_argument("HFV face references an invalid cell");
            }
            face.cells.push_back(incidence.face);
            cell_faces_[incidence.face].push_back({.face = face_index,
                                                   .normal_x = incidence.normal_x,
                                                   .normal_z = incidence.normal_z});
        }
        faces_.push_back(std::move(face));
    };

    for (const NonmatchingHorizontalFace& face : grid.horizontal_faces) {
        const NonmatchingColumnCell& upper = grid.cells[face.upper_cell];
        append_face({.x = 0.5 * (upper.x_left + upper.x_right), .z = face.z}, face.length,
                    FaceKind::internal,
                    {{face.upper_cell, 0.0, 1.0}, {face.lower_cell, 0.0, -1.0}});
    }
    for (const NonmatchingVerticalSubface& face : grid.vertical_subfaces) {
        append_face({.x = face.x, .z = face.center_z()}, face.length(), FaceKind::internal,
                    {{face.left_cell, 1.0, 0.0}, {face.right_cell, -1.0, 0.0}});
    }
    for (const std::size_t cell_index : grid.top_cells) {
        const NonmatchingColumnCell& cell = grid.cells[cell_index];
        append_face({.x = cell.center().x, .z = cell.z_top}, cell.width(),
                    FaceKind::surface_dirichlet, {{cell.index, 0.0, -1.0}});
    }
    for (const std::size_t cell_index : grid.bottom_cells) {
        const NonmatchingColumnCell& cell = grid.cells[cell_index];
        append_face({.x = cell.center().x, .z = cell.z_bottom}, cell.width(),
                    FaceKind::basal_neumann, {{cell.index, 0.0, 1.0}});
    }
    for (const NonmatchingVerticalBoundaryFace& face : grid.surface_vertical_faces) {
        append_face({.x = face.x, .z = face.center_z()}, face.length(), FaceKind::surface_dirichlet,
                    {{face.cell, face.outward_x, 0.0}});
    }
    for (const NonmatchingVerticalBoundaryFace& face : grid.basal_vertical_faces) {
        append_face({.x = face.x, .z = face.center_z()}, face.length(),
                    FaceKind::basal_step_neumann, {{face.cell, face.outward_x, 0.0}});
    }
    for (const std::size_t cell_index : grid.columns.front()) {
        const NonmatchingColumnCell& cell = grid.cells[cell_index];
        append_face({.x = cell.x_left, .z = cell.center().z}, cell.height(), FaceKind::left_neumann,
                    {{cell.index, -1.0, 0.0}});
    }
    for (const std::size_t cell_index : grid.columns.back()) {
        const NonmatchingColumnCell& cell = grid.cells[cell_index];
        append_face({.x = cell.x_right, .z = cell.center().z}, cell.height(),
                    FaceKind::right_neumann, {{cell.index, 1.0, 0.0}});
    }

    for (Face& face : faces_) {
        if (face.kind != FaceKind::surface_dirichlet) {
            face.unknown = grid.cells.size() + face_unknown_count_;
            ++face_unknown_count_;
        }
    }

    for (const NonmatchingColumnCell& cell : grid.cells) {
        double normal_x = 0.0;
        double normal_z = 0.0;
        double cone_area = 0.0;
        for (const CellFace& incidence : cell_faces_[cell.index]) {
            const Face& face = faces_[incidence.face];
            normal_x += face.length * incidence.normal_x;
            normal_z += face.length * incidence.normal_z;
            const double distance =
                std::abs((face.center.x - cell.center().x) * incidence.normal_x +
                         (face.center.z - cell.center().z) * incidence.normal_z);
            cone_area += 0.5 * face.length * distance;
        }
        const double scale = std::max({1.0, cell.width(), cell.height()});
        if (std::abs(normal_x) > 1e-10 * scale || std::abs(normal_z) > 1e-10 * scale ||
            std::abs(cone_area - cell.area()) > 1e-10 * std::max(1.0, cell.area())) {
            throw std::runtime_error("HFV local faces do not form a closed cell boundary");
        }
    }
}

Vector::Index NonmatchingColumnHfvSystem::size() const noexcept {
    return static_cast<Vector::Index>(grid_->cells.size() + face_unknown_count_);
}

std::size_t NonmatchingColumnHfvSystem::cell_count() const noexcept {
    return grid_->cells.size();
}

std::size_t NonmatchingColumnHfvSystem::face_unknown_count() const noexcept {
    return face_unknown_count_;
}

std::span<const double> NonmatchingColumnHfvSystem::thermal_conductivity() const noexcept {
    return thermal_conductivity_;
}

std::span<const double> NonmatchingColumnHfvSystem::heat_production() const noexcept {
    return heat_production_;
}

void NonmatchingColumnHfvSystem::update_properties(const Vector& solution) const {
    if (!property_updater_) {
        return;
    }
    property_updater_(std::span<const double>(solution.native().data(), grid_->cells.size()),
                      thermal_conductivity_, heat_production_);
    for (const double value : thermal_conductivity_) {
        if (!std::isfinite(value) || !(value > 0.0)) {
            throw std::runtime_error("HFV property update produced invalid conductivity");
        }
    }
}

double NonmatchingColumnHfvSystem::prescribed_face_value(const Face& face) const {
    if (face.kind != FaceKind::surface_dirichlet) {
        throw std::logic_error("Requested a prescribed value on a non-Dirichlet HFV face");
    }
    return boundary_data_.surface_temperature(face.center);
}

double NonmatchingColumnHfvSystem::prescribed_inward_flux(const Face& face) const {
    switch (face.kind) {
    case FaceKind::basal_neumann:
        return boundary_data_.basal_inward_flux(face.center);
    case FaceKind::left_neumann:
        return boundary_data_.left_inward_flux ? boundary_data_.left_inward_flux(face.center) : 0.0;
    case FaceKind::right_neumann:
        return boundary_data_.right_inward_flux ? boundary_data_.right_inward_flux(face.center)
                                                : 0.0;
    case FaceKind::basal_step_neumann:
        return 0.0;
    case FaceKind::internal:
    case FaceKind::surface_dirichlet:
        break;
    }
    throw std::logic_error("Requested a prescribed flux on an incompatible HFV face");
}

namespace {

template <typename LocalFaces, typename Faces>
LocalMatrix local_matrix(const NonmatchingColumnGrid& grid, const LocalFaces& local_faces,
                         const Faces& faces, const std::size_t cell_index,
                         const double conductivity) {
    const NonmatchingColumnCell& cell = grid.cells[cell_index];
    const Point2D center = cell.center();
    const Eigen::Index count = static_cast<Eigen::Index>(local_faces.size());
    LocalMatrix matrix = LocalMatrix::Zero(count, count);
    std::vector<Eigen::Vector2d> base(static_cast<std::size_t>(count));
    for (Eigen::Index j = 0; j < count; ++j) {
        const auto& incidence = local_faces[static_cast<std::size_t>(j)];
        const auto& face = faces[incidence.face];
        base[static_cast<std::size_t>(j)] =
            face.length / cell.area() * Eigen::Vector2d(incidence.normal_x, incidence.normal_z);
    }
    constexpr double stabilisation = 1.4142135623730950488;
    for (Eigen::Index tau = 0; tau < count; ++tau) {
        const auto& incidence = local_faces[static_cast<std::size_t>(tau)];
        const auto& face = faces[incidence.face];
        const Eigen::Vector2d normal(incidence.normal_x, incidence.normal_z);
        const Eigen::Vector2d offset(face.center.x - center.x, face.center.z - center.z);
        const double distance = std::abs(offset.dot(normal));
        if (!std::isfinite(distance) || !(distance > 0.0)) {
            throw std::runtime_error("HFV cell point is not strictly inside a face half-space");
        }
        const double cone_measure = 0.5 * face.length * distance;
        std::vector<Eigen::Vector2d> gradients(static_cast<std::size_t>(count));
        for (Eigen::Index j = 0; j < count; ++j) {
            const double kronecker = j == tau ? 1.0 : 0.0;
            const Eigen::Vector2d& consistent = base[static_cast<std::size_t>(j)];
            gradients[static_cast<std::size_t>(j)] =
                consistent +
                stabilisation / distance * (kronecker - consistent.dot(offset)) * normal;
        }
        for (Eigen::Index j = 0; j < count; ++j) {
            for (Eigen::Index k = 0; k < count; ++k) {
                matrix(j, k) += conductivity * cone_measure *
                                gradients[static_cast<std::size_t>(j)].dot(
                                    gradients[static_cast<std::size_t>(k)]);
            }
        }
    }
    return matrix;
}

} // namespace

void NonmatchingColumnHfvSystem::assemble_system(const Vector& linearization, SparseMatrix& matrix,
                                                 Vector& rhs) const {
    require_solution_size(linearization, size());
    update_properties(linearization);
    rhs.resize(size());
    rhs.set_zero();
    std::vector<MatrixTriplet> triplets;
    triplets.reserve(grid_->cells.size() * 36);

    for (const NonmatchingColumnCell& cell : grid_->cells) {
        const auto& local_faces = cell_faces_[cell.index];
        const LocalMatrix local = local_matrix(*grid_, local_faces, faces_, cell.index,
                                               thermal_conductivity_[cell.index]);
        rhs[static_cast<Vector::Index>(cell.index)] += heat_production_[cell.index] * cell.area();

        for (std::size_t j = 0; j < local_faces.size(); ++j) {
            const Face& row_face = faces_[local_faces[j].face];
            const std::array<std::pair<std::size_t, double>, 2> row_terms{{
                {cell.index, -1.0},
                {row_face.unknown.value_or(cell.index), row_face.unknown ? 1.0 : 0.0},
            }};
            for (std::size_t k = 0; k < local_faces.size(); ++k) {
                const Face& column_face = faces_[local_faces[k].face];
                const double coefficient =
                    local(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(k));
                const std::array<std::pair<std::size_t, double>, 2> column_terms{{
                    {cell.index, -1.0},
                    {column_face.unknown.value_or(cell.index), column_face.unknown ? 1.0 : 0.0},
                }};
                for (const auto& [row, row_weight] : row_terms) {
                    if (row_weight == 0.0) {
                        continue;
                    }
                    for (const auto& [column, column_weight] : column_terms) {
                        if (column_weight != 0.0) {
                            triplets.emplace_back(static_cast<Vector::Index>(row),
                                                  static_cast<Vector::Index>(column),
                                                  row_weight * coefficient * column_weight);
                        }
                    }
                    if (!column_face.unknown) {
                        rhs[static_cast<Vector::Index>(row)] -=
                            row_weight * coefficient * prescribed_face_value(column_face);
                    }
                }
            }
        }
    }

    for (const Face& face : faces_) {
        if (face.unknown && face.kind != FaceKind::internal) {
            rhs[static_cast<Vector::Index>(*face.unknown)] +=
                prescribed_inward_flux(face) * face.length;
        }
    }
    matrix = SparseMatrix(size(), size(), SparseStorageOrder::csc);
    matrix.set_from_triplets(triplets);
    matrix.make_compressed();
}

void NonmatchingColumnHfvSystem::assemble_residual(const Vector& solution, Vector& residual) const {
    SparseMatrix matrix;
    Vector rhs;
    assemble_system(solution, matrix, rhs);
    residual = matrix * solution - rhs;
}

Vector
NonmatchingColumnHfvSystem::interpolate(const std::function<double(Point2D)>& function) const {
    if (!function) {
        throw std::invalid_argument("HFV interpolation requires a function");
    }
    Vector result(size());
    for (const NonmatchingColumnCell& cell : grid_->cells) {
        result[static_cast<Vector::Index>(cell.index)] = function(cell.center());
    }
    for (const Face& face : faces_) {
        if (face.unknown) {
            result[static_cast<Vector::Index>(*face.unknown)] = function(face.center);
        }
    }
    return result;
}

NonmatchingHfvEnergyRates NonmatchingColumnHfvSystem::energy_rates(const Vector& solution) const {
    require_solution_size(solution, size());
    update_properties(solution);
    NonmatchingHfvEnergyRates result;
    for (const NonmatchingColumnCell& cell : grid_->cells) {
        result.heat_production += heat_production_[cell.index] * cell.area();
        const auto& local_faces = cell_faces_[cell.index];
        const LocalMatrix local = local_matrix(*grid_, local_faces, faces_, cell.index,
                                               thermal_conductivity_[cell.index]);
        Eigen::VectorXd difference(static_cast<Eigen::Index>(local_faces.size()));
        for (std::size_t j = 0; j < local_faces.size(); ++j) {
            const Face& face = faces_[local_faces[j].face];
            const double face_value = face.unknown
                                          ? solution[static_cast<Vector::Index>(*face.unknown)]
                                          : prescribed_face_value(face);
            difference[static_cast<Eigen::Index>(j)] =
                face_value - solution[static_cast<Vector::Index>(cell.index)];
        }
        const Eigen::VectorXd outward_flux = -local * difference;
        for (std::size_t j = 0; j < local_faces.size(); ++j) {
            const Face& face = faces_[local_faces[j].face];
            const double flux = outward_flux[static_cast<Eigen::Index>(j)];
            switch (face.kind) {
            case FaceKind::surface_dirichlet:
                result.surface_outflow += flux;
                break;
            case FaceKind::basal_neumann:
            case FaceKind::basal_step_neumann:
                result.basal_inflow -= flux;
                break;
            case FaceKind::left_neumann:
            case FaceKind::right_neumann:
                result.lateral_inflow -= flux;
                break;
            case FaceKind::internal:
                break;
            }
        }
    }
    return result;
}

NonmatchingHfvDiagnostics NonmatchingColumnHfvSystem::diagnostics(const Vector& solution) const {
    require_solution_size(solution, size());
    update_properties(solution);
    NonmatchingHfvDiagnostics result{.face_unknowns = face_unknown_count_};
    std::vector<double> internal_flux(faces_.size(), 0.0);
    for (const NonmatchingColumnCell& cell : grid_->cells) {
        const auto& local_faces = cell_faces_[cell.index];
        result.maximum_cell_faces = std::max(result.maximum_cell_faces, local_faces.size());
        const LocalMatrix local = local_matrix(*grid_, local_faces, faces_, cell.index,
                                               thermal_conductivity_[cell.index]);
        Eigen::VectorXd difference(static_cast<Eigen::Index>(local_faces.size()));
        for (std::size_t j = 0; j < local_faces.size(); ++j) {
            const Face& face = faces_[local_faces[j].face];
            const double face_value = face.unknown
                                          ? solution[static_cast<Vector::Index>(*face.unknown)]
                                          : prescribed_face_value(face);
            difference[static_cast<Eigen::Index>(j)] =
                face_value - solution[static_cast<Vector::Index>(cell.index)];
        }
        const Eigen::VectorXd outward_flux = -local * difference;
        for (std::size_t j = 0; j < local_faces.size(); ++j) {
            if (faces_[local_faces[j].face].kind == FaceKind::internal) {
                internal_flux[local_faces[j].face] += outward_flux[static_cast<Eigen::Index>(j)];
            }
        }
    }
    for (std::size_t i = 0; i < faces_.size(); ++i) {
        if (faces_[i].kind == FaceKind::internal) {
            result.internal_conservation_defect_linf =
                std::max(result.internal_conservation_defect_linf, std::abs(internal_flux[i]));
        }
    }
    return result;
}

double NonmatchingColumnHfvSystem::material_interface_outflow(const Vector& solution,
                                                              const int layer) const {
    require_solution_size(solution, size());
    update_properties(solution);
    double result = 0.0;
    for (const NonmatchingColumnCell& cell : grid_->cells) {
        if (cell.layer_id != layer) {
            continue;
        }
        const auto& local_faces = cell_faces_[cell.index];
        const LocalMatrix local = local_matrix(*grid_, local_faces, faces_, cell.index,
                                               thermal_conductivity_[cell.index]);
        Eigen::VectorXd difference(static_cast<Eigen::Index>(local_faces.size()));
        for (std::size_t j = 0; j < local_faces.size(); ++j) {
            const Face& face = faces_[local_faces[j].face];
            const double face_value = face.unknown
                                          ? solution[static_cast<Vector::Index>(*face.unknown)]
                                          : prescribed_face_value(face);
            difference[static_cast<Eigen::Index>(j)] =
                face_value - solution[static_cast<Vector::Index>(cell.index)];
        }
        const Eigen::VectorXd outward_flux = -local * difference;
        for (std::size_t j = 0; j < local_faces.size(); ++j) {
            const Face& face = faces_[local_faces[j].face];
            if (face.kind != FaceKind::internal || face.cells.size() != 2) {
                continue;
            }
            const std::size_t other = face.cells[0] == cell.index ? face.cells[1] : face.cells[0];
            if (grid_->cells[other].layer_id != layer) {
                result += outward_flux[static_cast<Eigen::Index>(j)];
            }
        }
    }
    return result;
}
