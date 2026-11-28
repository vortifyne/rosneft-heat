#include "basin_1d/geometry/basin_section_1d.hpp"
#include "basin_1d/inverse/section_observations.hpp"

#include <gtest/gtest.h>
#include <sstream>

TEST(SectionObservations, ProjectsSelectedWellsInsidePresentContacts) {
    const BasinSection1D section{50, {{10, 280, {}, {0, 100}}, {0, 285, {}, {10, 80}}}, {}};
    using enum BasinObservable;
    const std::vector<BasinObservation> points{{temperature, "A", 20, 10, 285, 1},
                                               {temperature, "B", 90, 10, 288, 2},
                                               {reflectance, "B", 90, 80, 0.4, 0.01},
                                               {temperature, "A", 20, 90, 300, 1},
                                               {temperature, "C", 40, 30, 295, 1}};
    const std::vector<std::string> wells{"A", "B"};
    std::ostringstream log;
    const auto selected = select_section_observations(section, points, wells, log);
    ASSERT_EQ(selected.size(), 3u);
    for (std::size_t i = 0; i < selected.size(); ++i) {
        EXPECT_EQ(selected[i].x, section.x);
        EXPECT_EQ(selected[i].well, points[i].well);
        EXPECT_EQ(selected[i].z, points[i].z);
        EXPECT_EQ(selected[i].kind, points[i].kind);
        EXPECT_EQ(selected[i].value, points[i].value);
        EXPECT_EQ(selected[i].error, points[i].error);
    }
    EXPECT_NE(log.str().find("3 of 4"), std::string::npos);
    EXPECT_TRUE(
        select_section_observations(section, std::span(points).subspan(3), wells, log).empty());
}
