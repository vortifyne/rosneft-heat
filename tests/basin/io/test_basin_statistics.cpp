#include "basin/forward/basin_forward_solver.hpp"
#include "basin/io/basin_statistics.hpp"

#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Row {
    std::size_t depth;
    double seconds;
    double percent;
};

std::vector<Row> timing_tree(const std::string& report) {
    std::istringstream input(report);
    std::vector<Row> rows;
    for (std::string line; std::getline(input, line);) {
        if (rows.empty() && !line.starts_with("Program"))
            continue;
        if (line.empty())
            break;
        std::istringstream fields(line);
        std::vector<std::string> columns;
        for (std::string column; fields >> column;)
            columns.push_back(column);
        rows.push_back({line.find_first_not_of(' '), std::stod(columns[columns.size() - 2]),
                        std::stod(columns.back())});
    }
    return rows;
}

} // namespace

TEST(BasinStatistics, ChildrenSumToParentsAndPercentagesUseProgramTime) {
    BasinForwardResult result;
    result.timesteps = 1;
    result.nonlinear_iterations = 3;
    result.linear_iterations = 12;
    result.timings = {.total_s = 80.0,
                      .grid_s = 10.0,
                      .transfer_s = 5.0,
                      .nonlinear = {.total_s = 50.0,
                                    .assembly = {.residual_s = 20.0, .matrix_s = 10.0},
                                    .linear_s = 15.0},
                      .properties = {.residual_s = 4.0, .matrix_s = 2.0},
                      .easy_ro_s = 5.0,
                      .output_s = 5.0};
    std::ostringstream output;
    write_basin_statistics(output, result, 100.0, 5.0, 5.0);
    const auto rows = timing_tree(output.str());
    ASSERT_FALSE(rows.empty());
    EXPECT_DOUBLE_EQ(rows.front().seconds, 100.0);
    EXPECT_DOUBLE_EQ(rows.front().percent, 100.0);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        EXPECT_DOUBLE_EQ(rows[i].percent, rows[i].seconds);
        double children = 0.0;
        bool has_children = false;
        for (std::size_t j = i + 1; j < rows.size() && rows[j].depth > rows[i].depth; ++j) {
            if (rows[j].depth == rows[i].depth + 2) {
                children += rows[j].seconds;
                has_children = true;
            }
        }
        if (has_children) {
            EXPECT_DOUBLE_EQ(children, rows[i].seconds);
        }
    }
    EXPECT_NE(output.str().find("Properties, total (included in assembly)"), std::string::npos);
    EXPECT_NE(output.str().find("Average nonlinear iterations per timestep: 3.00"),
              std::string::npos);
    EXPECT_NE(output.str().find("linear iterations per nonlinear iteration: 4.00"),
              std::string::npos);
}

TEST(BasinStatistics, EmptyCalculationHasFiniteZeroStatistics) {
    std::ostringstream output;
    write_basin_statistics(output, {}, 0.0, 0.0, 0.0);
    const auto rows = timing_tree(output.str());
    ASSERT_FALSE(rows.empty());
    for (const auto& row : rows) {
        EXPECT_DOUBLE_EQ(row.seconds, 0.0);
        EXPECT_DOUBLE_EQ(row.percent, 0.0);
    }
    EXPECT_EQ(output.str().find("nan"), std::string::npos);
    EXPECT_EQ(output.str().find("inf"), std::string::npos);
}
