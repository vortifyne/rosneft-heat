#include "basin/forward_output.hpp"

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace {

double minimum(const std::span<const double> field) {
    return *std::min_element(field.begin(), field.end());
}

double maximum(const std::span<const double> field) {
    return *std::max_element(field.begin(), field.end());
}

std::uintmax_t directory_size(const std::filesystem::path& directory) {
    std::uintmax_t result = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            result += entry.file_size();
        }
    }
    return result;
}

} // namespace

BasinForwardOutput::BasinForwardOutput(const std::filesystem::path& directory)
    : directory_(directory) {
    std::filesystem::create_directories(directory);
    statistics_.open(directory / "statistics.csv");
    epochs_.open(directory / "epochs.csv");
    transitions_.open(directory / "transitions.csv");
    energy_balance_.open(directory / "energy-balance.csv");
    if (!statistics_ || !epochs_ || !transitions_ || !energy_balance_) {
        throw std::runtime_error("Cannot open forward-solver tabular output");
    }
    statistics_ << "elapsed_ma,age_ma,kind,cells,nonlinear_iterations,cell_area_min,"
                   "temperature_min,temperature_max,Ro_min,Ro_max\n";
    epochs_ << "start_age_ma,end_age_ma,steps,nonlinear_iterations,linear_iterations,"
               "start_cells,end_cells,topology_regularized,wall_seconds\n";
    transitions_ << "age_ma,old_cells,new_cells\n";
    energy_balance_ << "elapsed_ma,age_ma,kind,relative_error\n";
}

void BasinForwardOutput::save_state(BasinState& state, const EasyRoModel& easy_ro,
                                    const double elapsed_ma, const double age_ma,
                                    const std::string& kind, const int nonlinear_iterations) {
    update_thermophysical_properties(vector_values(state.temperature), state.fields);
    for (const TQMesh::Facet* cell : state.mesh.cells()) {
        state.fields.cell_area[static_cast<std::size_t>(cell->index())] = cell->area();
    }
    std::vector<double> reflectance(state.mesh.cells().size());
    easy_ro.reflectance(state.maturity, reflectance);
    std::ostringstream name;
    name << "state_" << std::setw(4) << std::setfill('0') << index_++ << '_' << kind << ".vtu";
    const std::string filename = name.str();
    const std::vector<VtuCellField> fields = {
        {"temperature", vector_values(state.temperature)},
        {"porosity", state.fields.porosity},
        {"material_eta", state.fields.material_eta},
        {"cell_area", state.fields.cell_area},
        {"layer_id", state.fields.layer_id},
        {"lithotype", state.fields.lithotype_code},
        {"thermal_conductivity", state.fields.conductivity},
        {"volumetric_heat_capacity", state.fields.heat_capacity},
        {"heat_production", state.fields.heat_production},
        {"velocity_z", state.fields.velocity_z},
        {"Ro", reflectance},
    };
    write_vtu(state.mesh, directory_ / filename, fields);
    series_.push_back({elapsed_ma, filename});
    statistics_ << std::setprecision(15) << elapsed_ma << ',' << age_ma << ',' << kind << ','
                << state.mesh.cells().size() << ',' << nonlinear_iterations << ','
                << minimum(state.fields.cell_area) << ','
                << minimum(vector_values(state.temperature)) << ','
                << maximum(vector_values(state.temperature)) << ',' << minimum(reflectance) << ','
                << maximum(reflectance) << '\n';
}

void BasinForwardOutput::save_energy_balance(const double elapsed_ma, const double age_ma,
                                             const std::string& kind, const double relative_error) {
    energy_balance_ << std::setprecision(17) << elapsed_ma << ',' << age_ma << ',' << kind << ','
                    << relative_error << '\n';
}

void BasinForwardOutput::save_transition(const double age_ma, const std::size_t old_cells,
                                         const std::size_t new_cells) {
    transitions_ << std::setprecision(17) << age_ma << ',' << old_cells << ',' << new_cells << '\n';
}

void BasinForwardOutput::save_epoch(const double start_age_ma, const double end_age_ma,
                                    const int steps, const int nonlinear_iterations,
                                    const int linear_iterations, const std::size_t start_cells,
                                    const std::size_t end_cells, const bool topology_regularized,
                                    const double wall_seconds) {
    epochs_ << std::setprecision(17) << start_age_ma << ',' << end_age_ma << ',' << steps << ','
            << nonlinear_iterations << ',' << linear_iterations << ',' << start_cells << ','
            << end_cells << ',' << topology_regularized << ',' << wall_seconds << '\n';
}

void BasinForwardOutput::finish(const BasinForwardResult& result) {
    write_pvd(directory_ / "temperature.pvd", series_);
    statistics_.flush();
    epochs_.flush();
    transitions_.flush();
    energy_balance_.flush();
    const std::uintmax_t output_bytes = directory_size(directory_);
    std::ofstream summary(directory_ / "summary.csv");
    summary << "configurations,accepted_steps,nonlinear_iterations,"
               "maximum_nonlinear_iterations_per_step,linear_iterations,final_age_ma,"
               "topology_regularized_epochs,global_energy_balance,cells_min,cells_max,"
               "cell_diameter_max,wall_seconds,output_bytes\n"
            << result.configurations << ',' << result.accepted_steps << ','
            << result.nonlinear_iterations << ',' << result.maximum_nonlinear_iterations_per_step
            << ',' << result.linear_iterations << ',' << result.final_age_ma << ','
            << result.topology_regularized_epochs << ',' << result.global_energy_balance << ','
            << result.minimum_cells << ',' << result.maximum_cells << ','
            << result.maximum_cell_diameter << ',' << result.wall_seconds << ',' << output_bytes
            << '\n';
}
