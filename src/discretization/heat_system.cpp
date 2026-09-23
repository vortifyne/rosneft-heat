#include "discretization/heat_system.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace {

Eigen::Vector2d vector(const CppUtils::Vec2d& value) {
    return {value.x, value.y};
}

Point2D point(const CppUtils::Vec2d& value) {
    return {value.x, value.y};
}

void check_finite(const double value, const char* message) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(message);
    }
}

void check_field(std::span<const double> field, const std::size_t expected_size,
                 const bool require_positive, const char* message) {
    if (field.size() != expected_size) {
        throw std::invalid_argument("Heat-system field size does not match the mesh");
    }
    for (const double value : field) {
        if (!std::isfinite(value) || (require_positive && !(value > 0.0))) {
            throw std::invalid_argument(message);
        }
    }
}

} // namespace

struct HeatSystem::ResidualAssembly {
    Vector& residual;
};

struct HeatSystem::MatrixAssembly {
    std::vector<double>& diagonal;
    std::vector<MatrixTriplet>& triplets;
    double derivative_shift;
};

HeatSystem::HeatSystem(const Mesh2D& mesh, const std::span<const double> thermal_conductivity,
                       const std::span<const double> volumetric_heat_capacity,
                       const std::span<const double> heat_production,
                       HeatBoundaryConditions boundary_conditions)
    : thermal_conductivity_(thermal_conductivity.begin(), thermal_conductivity.end()),
      volumetric_heat_capacity_(volumetric_heat_capacity.begin(), volumetric_heat_capacity.end()),
      heat_production_(heat_production.begin(), heat_production.end()),
      boundary_conditions_(std::move(boundary_conditions)) {
    const std::size_t cell_count = mesh.cells().size();
    check_field(thermal_conductivity, cell_count, true,
                "Thermal conductivity must be finite and positive");
    check_field(volumetric_heat_capacity, cell_count, true,
                "Volumetric heat capacity must be finite and positive");
    check_field(heat_production, cell_count, false, "Heat production must be finite");
    if (!boundary_conditions_.surface_temperature || !boundary_conditions_.basal_heat_flux) {
        throw std::invalid_argument("Heat-system boundary functions must be set");
    }

    inverse_cell_area_.resize(cell_count);
    for (const TQMesh::Facet* cell : mesh.cells()) {
        const std::size_t index = static_cast<std::size_t>(cell->index());
        if (!(cell->area() > 0.0) || !std::isfinite(cell->area())) {
            throw std::invalid_argument("Heat-system cell area must be finite and positive");
        }
        inverse_cell_area_.at(index) = 1.0 / cell->area();
    }

    std::vector<LeastSquaresConnection> connections;
    connections.reserve(mesh.internal_edges().size());
    internal_faces_.reserve(mesh.internal_edges().size());
    for (const TQMesh::Edge* edge : mesh.internal_edges()) {
        const TQMesh::Facet& owner = mesh.owner(*edge);
        const TQMesh::Facet& neighbor = mesh.neighbor(*edge);
        const Eigen::Vector2d displacement = vector(neighbor.xy() - owner.xy());
        const Eigen::Vector2d normal = vector(mesh.normal_from_owner(*edge));
        const Eigen::Vector2d owner_to_face = vector(edge->xy() - owner.xy());
        const Eigen::Vector2d face_to_neighbor = vector(neighbor.xy() - edge->xy());
        const double normal_distance = normal.dot(displacement);
        const double owner_distance = std::abs(normal.dot(owner_to_face));
        const double neighbor_distance = std::abs(normal.dot(face_to_neighbor));
        if (!(normal_distance > 0.0) || !(owner_distance > 0.0) || !(neighbor_distance > 0.0)) {
            throw std::runtime_error("Invalid internal-face geometry for heat discretization");
        }

        const std::size_t owner_index = static_cast<std::size_t>(owner.index());
        const std::size_t neighbor_index = static_cast<std::size_t>(neighbor.index());
        internal_faces_.push_back({.owner = owner_index,
                                   .neighbor = neighbor_index,
                                   .length = edge->length(),
                                   .normal_distance = normal_distance,
                                   .owner_distance = owner_distance,
                                   .neighbor_distance = neighbor_distance,
                                   .correction = normal - (displacement / normal_distance)});
        connections.push_back({owner_index, neighbor_index, displacement});
    }

    std::vector<LeastSquaresBoundarySample> surface_samples;
    surface_samples.reserve(mesh.boundary_edges(BoundaryKind::top).size());
    surface_faces_.reserve(mesh.boundary_edges(BoundaryKind::top).size());
    for (const TQMesh::Edge* edge : mesh.boundary_edges(BoundaryKind::top)) {
        const TQMesh::Facet& cell = mesh.owner(*edge);
        const Eigen::Vector2d displacement = vector(edge->xy() - cell.xy());
        const Eigen::Vector2d normal = vector(mesh.normal_from_owner(*edge));
        const double normal_distance = normal.dot(displacement);
        if (!(normal_distance > 0.0)) {
            throw std::runtime_error("Invalid surface-face geometry for heat discretization");
        }
        const std::size_t cell_index = static_cast<std::size_t>(cell.index());
        surface_faces_.push_back({.cell = cell_index,
                                  .center = point(edge->xy()),
                                  .length = edge->length(),
                                  .normal_distance = normal_distance,
                                  .correction = normal - (displacement / normal_distance)});
        surface_samples.push_back({cell_index, displacement});
    }

    basal_faces_.reserve(mesh.boundary_edges(BoundaryKind::bottom).size());
    for (const TQMesh::Edge* edge : mesh.boundary_edges(BoundaryKind::bottom)) {
        const TQMesh::Facet& cell = mesh.owner(*edge);
        const Eigen::Vector2d displacement = vector(edge->xy() - cell.xy());
        const Eigen::Vector2d normal = vector(mesh.normal_from_owner(*edge));
        const double normal_distance = normal.dot(displacement);
        if (!(normal_distance > 0.0)) {
            throw std::runtime_error("Invalid basal-face geometry for heat discretization");
        }
        basal_faces_.push_back({.cell = static_cast<std::size_t>(cell.index()),
                                .center = point(edge->xy()),
                                .length = edge->length(),
                                .normal_distance = normal_distance,
                                .correction = normal - (displacement / normal_distance)});
    }
    lateral_face_count_ = mesh.boundary_edges(BoundaryKind::left).size() +
                          mesh.boundary_edges(BoundaryKind::right).size();

    gradient_reconstruction_.emplace(cell_count, connections, surface_samples);
    surface_temperatures_.resize(surface_faces_.size());
    gradients_.resize(cell_count);
}

Vector::Index HeatSystem::size() const noexcept {
    return static_cast<Vector::Index>(inverse_cell_area_.size());
}

void HeatSystem::assemble_residual(const double time, const Vector& solution,
                                   const Vector& solution_derivative, Vector& residual) const {
    check_finite(time, "Assembly time must be finite");
    check_vector_sizes(solution, solution_derivative);
    prepare_gradients(time, solution);

    residual.resize(size());
    residual.set_zero();
    ResidualAssembly assembly{residual};
    assemble(assembly, time, solution, solution_derivative);
}

void HeatSystem::assemble_matrix(const NonlinearMethod method, const double time,
                                 const Vector& solution, const Vector& solution_derivative,
                                 const double derivative_shift, SparseMatrix& matrix) const {
    if (method != NonlinearMethod::picard) {
        throw std::invalid_argument("HeatSystem supports only the Picard matrix");
    }
    check_finite(time, "Assembly time must be finite");
    check_finite(derivative_shift, "Derivative shift must be finite");
    check_vector_sizes(solution, solution_derivative);

    std::vector<double> diagonal(static_cast<std::size_t>(size()), 0.0);
    std::vector<MatrixTriplet> triplets;
    triplets.reserve(static_cast<std::size_t>(size()) + (2 * internal_faces_.size()));
    MatrixAssembly assembly{diagonal, triplets, derivative_shift};
    assemble(assembly, time, solution, solution_derivative);
    for (Vector::Index index = 0; index < size(); ++index) {
        triplets.emplace_back(index, index, diagonal[static_cast<std::size_t>(index)]);
    }

    matrix.resize(size(), size());
    matrix.set_from_triplets(triplets);
}

template <typename Assembly>
void HeatSystem::assemble(Assembly& assembly, const double time, const Vector& solution,
                          const Vector& solution_derivative) const {
    assemble_accumulation(assembly, time, solution, solution_derivative);
    assemble_internal_diffusion(assembly, time, solution, solution_derivative);
    assemble_source(assembly, time, solution, solution_derivative);
    assemble_surface_boundary(assembly, time, solution, solution_derivative);
    assemble_basal_boundary(assembly, time, solution, solution_derivative);
    assemble_lateral_boundaries(assembly, time, solution, solution_derivative);
}

void HeatSystem::assemble_accumulation(ResidualAssembly& assembly, const double time,
                                       const Vector& solution,
                                       const Vector& solution_derivative) const {
    static_cast<void>(time);
    static_cast<void>(solution);
    for (Vector::Index index = 0; index < size(); ++index) {
        assembly.residual[index] +=
            volumetric_heat_capacity_[static_cast<std::size_t>(index)] * solution_derivative[index];
    }
}

void HeatSystem::assemble_accumulation(MatrixAssembly& assembly, const double time,
                                       const Vector& solution,
                                       const Vector& solution_derivative) const {
    static_cast<void>(time);
    static_cast<void>(solution);
    static_cast<void>(solution_derivative);
    for (std::size_t index = 0; index < assembly.diagonal.size(); ++index) {
        assembly.diagonal[index] += volumetric_heat_capacity_[index] * assembly.derivative_shift;
    }
}

void HeatSystem::assemble_internal_diffusion(ResidualAssembly& assembly, const double time,
                                             const Vector& solution,
                                             const Vector& solution_derivative) const {
    static_cast<void>(time);
    static_cast<void>(solution_derivative);
    assemble_orthogonal_diffusion_residual(assembly, solution);
    assemble_nonorthogonal_correction_residual(assembly);
}

void HeatSystem::assemble_internal_diffusion(MatrixAssembly& assembly, const double time,
                                             const Vector& solution,
                                             const Vector& solution_derivative) const {
    static_cast<void>(time);
    static_cast<void>(solution);
    static_cast<void>(solution_derivative);
    assemble_orthogonal_diffusion_matrix(assembly);
    // An implicit correction can be added here with one independent call:
    // assemble_nonorthogonal_correction_matrix(assembly);
}

void HeatSystem::assemble_source(ResidualAssembly& assembly, const double time,
                                 const Vector& solution, const Vector& solution_derivative) const {
    static_cast<void>(time);
    static_cast<void>(solution);
    static_cast<void>(solution_derivative);
    for (Vector::Index index = 0; index < size(); ++index) {
        assembly.residual[index] -= heat_production_[static_cast<std::size_t>(index)];
    }
}

void HeatSystem::assemble_source(MatrixAssembly& assembly, const double time,
                                 const Vector& solution, const Vector& solution_derivative) const {
    static_cast<void>(assembly);
    static_cast<void>(time);
    static_cast<void>(solution);
    static_cast<void>(solution_derivative);
}

void HeatSystem::assemble_surface_boundary(ResidualAssembly& assembly, const double time,
                                           const Vector& solution,
                                           const Vector& solution_derivative) const {
    static_cast<void>(time);
    static_cast<void>(solution_derivative);
    for (std::size_t index = 0; index < surface_faces_.size(); ++index) {
        const BoundaryFace& face = surface_faces_[index];
        const double conductivity = thermal_conductivity_[face.cell];
        const double coefficient = conductivity * face.length / face.normal_distance;
        const double orthogonal_flux =
            coefficient *
            (solution[static_cast<Vector::Index>(face.cell)] - surface_temperatures_[index]);
        const double correction_flux =
            -conductivity * face.length * gradients_[face.cell].dot(face.correction);
        assembly.residual[static_cast<Vector::Index>(face.cell)] +=
            (orthogonal_flux + correction_flux) * inverse_cell_area_[face.cell];
    }
}

void HeatSystem::assemble_surface_boundary(MatrixAssembly& assembly, const double time,
                                           const Vector& solution,
                                           const Vector& solution_derivative) const {
    static_cast<void>(time);
    static_cast<void>(solution);
    static_cast<void>(solution_derivative);
    for (const BoundaryFace& face : surface_faces_) {
        const double coefficient = thermal_conductivity_[face.cell] * face.length /
                                   face.normal_distance * inverse_cell_area_[face.cell];
        assembly.diagonal[face.cell] += coefficient;
    }
}

void HeatSystem::assemble_basal_boundary(ResidualAssembly& assembly, const double time,
                                         const Vector& solution,
                                         const Vector& solution_derivative) const {
    static_cast<void>(solution);
    static_cast<void>(solution_derivative);
    for (const BoundaryFace& face : basal_faces_) {
        const double flux = boundary_conditions_.basal_heat_flux(face.center, time);
        check_finite(flux, "Basal heat flux must be finite");
        assembly.residual[static_cast<Vector::Index>(face.cell)] -=
            flux * face.length * inverse_cell_area_[face.cell];
    }
}

void HeatSystem::assemble_basal_boundary(MatrixAssembly& assembly, const double time,
                                         const Vector& solution,
                                         const Vector& solution_derivative) const {
    static_cast<void>(assembly);
    static_cast<void>(time);
    static_cast<void>(solution);
    static_cast<void>(solution_derivative);
}

void HeatSystem::assemble_lateral_boundaries(ResidualAssembly& assembly, const double time,
                                             const Vector& solution,
                                             const Vector& solution_derivative) const {
    static_cast<void>(assembly);
    static_cast<void>(time);
    static_cast<void>(solution);
    static_cast<void>(solution_derivative);
    static_cast<void>(lateral_face_count_);
}

void HeatSystem::assemble_lateral_boundaries(MatrixAssembly& assembly, const double time,
                                             const Vector& solution,
                                             const Vector& solution_derivative) const {
    static_cast<void>(assembly);
    static_cast<void>(time);
    static_cast<void>(solution);
    static_cast<void>(solution_derivative);
    static_cast<void>(lateral_face_count_);
}

void HeatSystem::assemble_orthogonal_diffusion_residual(ResidualAssembly& assembly,
                                                        const Vector& solution) const {
    for (const InternalFace& face : internal_faces_) {
        const double coefficient = face_conductivity(face) * face.length / face.normal_distance;
        const double flux = coefficient * (solution[static_cast<Vector::Index>(face.owner)] -
                                           solution[static_cast<Vector::Index>(face.neighbor)]);
        assembly.residual[static_cast<Vector::Index>(face.owner)] +=
            flux * inverse_cell_area_[face.owner];
        assembly.residual[static_cast<Vector::Index>(face.neighbor)] -=
            flux * inverse_cell_area_[face.neighbor];
    }
}

void HeatSystem::assemble_nonorthogonal_correction_residual(ResidualAssembly& assembly) const {
    for (const InternalFace& face : internal_faces_) {
        const double total_distance = face.owner_distance + face.neighbor_distance;
        const auto face_gradient =
            (face.neighbor_distance / total_distance) * gradients_[face.owner] +
            (face.owner_distance / total_distance) * gradients_[face.neighbor];
        const double flux =
            -face_conductivity(face) * face.length * face_gradient.dot(face.correction);
        assembly.residual[static_cast<Vector::Index>(face.owner)] +=
            flux * inverse_cell_area_[face.owner];
        assembly.residual[static_cast<Vector::Index>(face.neighbor)] -=
            flux * inverse_cell_area_[face.neighbor];
    }
}

void HeatSystem::assemble_orthogonal_diffusion_matrix(MatrixAssembly& assembly) const {
    for (const InternalFace& face : internal_faces_) {
        const double coefficient = face_conductivity(face) * face.length / face.normal_distance;
        const double owner_coefficient = coefficient * inverse_cell_area_[face.owner];
        const double neighbor_coefficient = coefficient * inverse_cell_area_[face.neighbor];
        assembly.diagonal[face.owner] += owner_coefficient;
        assembly.diagonal[face.neighbor] += neighbor_coefficient;
        assembly.triplets.emplace_back(static_cast<Vector::Index>(face.owner),
                                       static_cast<Vector::Index>(face.neighbor),
                                       -owner_coefficient);
        assembly.triplets.emplace_back(static_cast<Vector::Index>(face.neighbor),
                                       static_cast<Vector::Index>(face.owner),
                                       -neighbor_coefficient);
    }
}

double HeatSystem::face_conductivity(const InternalFace& face) const noexcept {
    const double owner_conductivity = thermal_conductivity_[face.owner];
    const double neighbor_conductivity = thermal_conductivity_[face.neighbor];
    return (face.owner_distance + face.neighbor_distance) /
           ((face.owner_distance / owner_conductivity) +
            (face.neighbor_distance / neighbor_conductivity));
}

void HeatSystem::check_vector_sizes(const Vector& solution,
                                    const Vector& solution_derivative) const {
    if (solution.size() != size() || solution_derivative.size() != size()) {
        throw std::invalid_argument("Heat-system vector sizes must match the mesh cell count");
    }
}

void HeatSystem::prepare_gradients(const double time, const Vector& solution) const {
    for (std::size_t index = 0; index < surface_faces_.size(); ++index) {
        surface_temperatures_[index] =
            boundary_conditions_.surface_temperature(surface_faces_[index].center, time);
        check_finite(surface_temperatures_[index], "Surface temperature must be finite");
    }
    gradient_reconstruction_->reconstruct(
        std::span<const double>(solution.native().data(),
                                static_cast<std::size_t>(solution.size())),
        surface_temperatures_, gradients_);
}
