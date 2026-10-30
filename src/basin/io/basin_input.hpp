#pragma once

#include "basin/basin_problem.hpp"
#include "basin/physics/thermophysical_properties.hpp"

#include <filesystem>
#include <span>
#include <unordered_map>
#include <vector>

class BasinInput {
public:
    static BasinInput read(const std::filesystem::path& directory);

    [[nodiscard]] const LithotypeThermophysicalProperties& lithotype(int code) const;
    [[nodiscard]] std::span<const BasinConfiguration> configurations() const noexcept;
    [[nodiscard]] const EasyRoInput& easy_ro() const noexcept;

private:
    std::vector<BasinConfiguration> configurations_;
    std::unordered_map<int, LithotypeThermophysicalProperties> lithotypes_;
    EasyRoInput easy_ro_;
};
