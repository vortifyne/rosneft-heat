#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Cell {
    int material = -1;
    double top = 0.0;
    double bottom = 0.0;
    double temperature = 0.0;
};

struct Column {
    double left = 0.0;
    double right = 0.0;
    std::vector<Cell> cells;
};

struct Field {
    std::vector<Column> columns;
    double area = 0.0;
};

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::istringstream stream(line);
    for (std::string field; std::getline(stream, field, ',');) {
        fields.push_back(std::move(field));
    }
    return fields;
}

Field read_field(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open " + path);
    }
    std::string line;
    if (!std::getline(input, line) || line !=
            "index,column,layer,x_left,x_right,z_top,z_bottom,temperature") {
        throw std::runtime_error("Unexpected cells.csv header in " + path);
    }

    Field result;
    std::size_t previous_column = std::numeric_limits<std::size_t>::max();
    std::size_t expected_column = 0;
    while (std::getline(input, line)) {
        const std::vector<std::string> fields = split_csv_line(line);
        if (fields.size() != 8) {
            throw std::runtime_error("Malformed cells.csv row in " + path);
        }
        const std::size_t column_index = std::stoull(fields[1]);
        const int material = std::stoi(fields[2]);
        const double left = std::stod(fields[3]);
        const double right = std::stod(fields[4]);
        const double top = std::stod(fields[5]);
        const double bottom = std::stod(fields[6]);
        const double temperature = std::stod(fields[7]);
        if (!(right > left) || !(bottom > top) || !std::isfinite(temperature)) {
            throw std::runtime_error("Invalid cell geometry or temperature in " + path);
        }
        if (column_index != previous_column) {
            if (column_index != expected_column) {
                throw std::runtime_error("Non-consecutive column index in " + path);
            }
            result.columns.push_back({left, right, {}});
            previous_column = column_index;
            ++expected_column;
        }
        Column& column = result.columns.back();
        const double scale = std::max({1.0, std::abs(left), std::abs(right)});
        if (std::abs(column.left - left) > 1e-12 * scale ||
            std::abs(column.right - right) > 1e-12 * scale) {
            throw std::runtime_error("Inconsistent x interval within column in " + path);
        }
        column.cells.push_back({material, top, bottom, temperature});
        result.area += (right - left) * (bottom - top);
    }
    if (result.columns.empty()) {
        throw std::runtime_error("Empty cell field in " + path);
    }
    for (Column& column : result.columns) {
        std::sort(column.cells.begin(), column.cells.end(),
                  [](const Cell& first, const Cell& second) { return first.top < second.top; });
    }
    return result;
}

struct Comparison {
    std::size_t intersections = 0;
    double overlap_area = 0.0;
    double mismatch_area = 0.0;
    double squared_difference = 0.0;
    double absolute_difference = 0.0;
    double signed_difference = 0.0;
    double first_integral = 0.0;
    double second_integral = 0.0;
    double maximum_difference = 0.0;
};

void compare_columns(const Column& first, const Column& second, const double width,
                     Comparison& result) {
    std::size_t i = 0;
    std::size_t j = 0;
    while (i < first.cells.size() && j < second.cells.size()) {
        const Cell& a = first.cells[i];
        const Cell& b = second.cells[j];
        const double top = std::max(a.top, b.top);
        const double bottom = std::min(a.bottom, b.bottom);
        if (bottom > top) {
            const double area = width * (bottom - top);
            const double difference = a.temperature - b.temperature;
            result.overlap_area += area;
            result.squared_difference += area * difference * difference;
            result.absolute_difference += area * std::abs(difference);
            result.signed_difference += area * difference;
            result.first_integral += area * a.temperature;
            result.second_integral += area * b.temperature;
            result.maximum_difference = std::max(result.maximum_difference, std::abs(difference));
            if (a.material != b.material) {
                result.mismatch_area += area;
            }
            ++result.intersections;
        }
        if (a.bottom < b.bottom) {
            ++i;
        } else if (b.bottom < a.bottom) {
            ++j;
        } else {
            ++i;
            ++j;
        }
    }
}

Comparison compare(const Field& first, const Field& second) {
    Comparison result;
    std::size_t i = 0;
    std::size_t j = 0;
    while (i < first.columns.size() && j < second.columns.size()) {
        const Column& a = first.columns[i];
        const Column& b = second.columns[j];
        const double left = std::max(a.left, b.left);
        const double right = std::min(a.right, b.right);
        if (right > left) {
            compare_columns(a, b, right - left, result);
        }
        if (a.right < b.right) {
            ++i;
        } else if (b.right < a.right) {
            ++j;
        } else {
            ++i;
            ++j;
        }
    }
    if (!(result.overlap_area > 0.0)) {
        throw std::runtime_error("The fields have no common area");
    }
    return result;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::invalid_argument(
                "usage: compare_nonmatching_cell_fields FIRST_CELLS.csv SECOND_CELLS.csv");
        }
        const Field first = read_field(argv[1]);
        const Field second = read_field(argv[2]);
        const Comparison result = compare(first, second);
        const double inverse_area = 1.0 / result.overlap_area;
        const double symmetric_difference =
            std::max(0.0, first.area + second.area - 2.0 * result.overlap_area);
        std::cout << "intersections,area_first,area_second,overlap_area,symmetric_difference_area,"
                     "material_mismatch_area,material_mismatch_fraction,rms,mean_abs,mean_signed,"
                     "max_abs,mean_first,mean_second\n"
                  << std::setprecision(17) << result.intersections << ',' << first.area << ','
                  << second.area << ',' << result.overlap_area << ','
                  << symmetric_difference << ','
                  << result.mismatch_area << ',' << result.mismatch_area * inverse_area << ','
                  << std::sqrt(result.squared_difference * inverse_area) << ','
                  << result.absolute_difference * inverse_area << ','
                  << result.signed_difference * inverse_area << ',' << result.maximum_difference
                  << ',' << result.first_integral * inverse_area << ','
                  << result.second_integral * inverse_area << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
