#pragma once

#include "basin/basin_forward_solver.hpp"
#include "basin/basin_state.hpp"
#include "basin/vtu_output.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

class BasinForwardOutput {
public:
    explicit BasinForwardOutput(const std::filesystem::path& directory);

    void save_state(BasinState& state, const EasyRoModel& easy_ro, double elapsed_ma, double age_ma,
                    const std::string& kind, int nonlinear_iterations);
    void save_energy_balance(double elapsed_ma, double age_ma, const std::string& kind,
                             double relative_error);
    void save_transition(double age_ma, std::size_t old_cells, std::size_t new_cells);
    void save_epoch(double start_age_ma, double end_age_ma, int steps, int nonlinear_iterations,
                    int linear_iterations, std::size_t start_cells, std::size_t end_cells,
                    bool topology_regularized, double wall_seconds);
    void finish(const BasinForwardResult& result);

private:
    std::filesystem::path directory_;
    std::ofstream statistics_;
    std::ofstream epochs_;
    std::ofstream transitions_;
    std::ofstream energy_balance_;
    std::vector<PvdEntry> series_;
    int index_ = 0;
};
