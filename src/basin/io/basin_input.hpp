#pragma once

#include "basin/basin_problem.hpp"
#include "basin/physics/easy_ro.hpp"
#include "basin/physics/thermal_properties.hpp"

#include <filesystem>
#include <span>
#include <unordered_map>
#include <vector>

class BasinInput {
public:
    static BasinInput read(const std::filesystem::path& directory);

    [[nodiscard]] const LithotypeProperties& lithotype(int code) const;
    [[nodiscard]] std::span<const BasinConfig> configs() const noexcept;
    [[nodiscard]] const EasyRoParameters& easy_ro() const noexcept;

private:
    std::vector<BasinConfig> configs_;
    std::unordered_map<int, LithotypeProperties> lithotypes_;
    EasyRoParameters easy_ro_{};
};
