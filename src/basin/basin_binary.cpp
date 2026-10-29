#include "basin/basin_binary.hpp"

#include <bit>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {

template <typename T> void write_value(std::ofstream& file, const T& value) {
    file.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T> void write_array(std::ofstream& file, const std::vector<T>& values) {
    file.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(T)));
}

} // namespace

void write_basin_binary(const std::filesystem::path& path, const BasinForwardSnapshot& snapshot) {
    static_assert(std::endian::native == std::endian::little);
    const std::size_t count = snapshot.grid.cells.size();
    if (snapshot.temperature.size() != count || snapshot.reflectance_percent.size() != count ||
        (count > 0 && snapshot.maturity_integrals.size() % count != 0)) {
        throw std::invalid_argument("column-grid binary field sizes do not match");
    }
    std::ofstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot create column-grid binary snapshot");
    file.write("CGRID001", 8);
    write_value(file, static_cast<std::uint32_t>(snapshot.kind));
    write_value(file, snapshot.age_ma);
    write_value(file, static_cast<std::uint64_t>(snapshot.grid.x.size()));
    write_value(file, static_cast<std::uint64_t>(count));
    write_value(file, static_cast<std::uint32_t>(
                          count == 0 ? 0 : snapshot.maturity_integrals.size() / count));
    write_array(file, snapshot.grid.x);
    std::vector<std::uint64_t> column(count);
    std::vector<std::int32_t> layer_id(count);
    std::vector<double> z_top(count), z_bottom(count);
    for (const auto& cell : snapshot.grid.cells) {
        column[cell.index] = static_cast<std::uint64_t>(cell.column);
        layer_id[cell.index] = static_cast<std::int32_t>(cell.layer_id);
        z_top[cell.index] = cell.z_top;
        z_bottom[cell.index] = cell.z_bottom;
    }
    write_array(file, column);
    write_array(file, layer_id);
    write_array(file, z_top);
    write_array(file, z_bottom);
    file.write(reinterpret_cast<const char*>(snapshot.temperature.data()),
               static_cast<std::streamsize>(count * sizeof(double)));
    file.write(reinterpret_cast<const char*>(snapshot.reflectance_percent.data()),
               static_cast<std::streamsize>(count * sizeof(double)));
    file.write(reinterpret_cast<const char*>(snapshot.maturity_integrals.data()),
               static_cast<std::streamsize>(snapshot.maturity_integrals.size() * sizeof(double)));
    if (!file)
        throw std::runtime_error("Failed writing column-grid binary snapshot");
}
