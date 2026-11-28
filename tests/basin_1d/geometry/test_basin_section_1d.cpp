#include "basin_1d/geometry/basin_section_1d.hpp"

#include <array>
#include <gtest/gtest.h>

TEST(BasinSection1D, SamplesProfilesAndUnfoldsTheNewLayerAtFixedX) {
    BasinLayerProfile base{2, "base", {0, 4}, {0, 0}, {8, 16}, {2, 3}, {0.2, 0.6}, {0.1, 0.3}};
    BasinConfig old{10, {base}, {{0, 4}, {280, 288}}, {}};
    BasinLayerProfile cap{1, "cap", {0, 4}, {0, 0}, {4, 4}, {1, 1}, {0.5, 0.5}, {0.4, 0.4}};
    base.top = {4, 4};
    base.bottom = {12, 20};
    BasinConfig young{0, {cap, base}, old.T_surf, {}};

    const auto section = make_basin_section_1d(std::array{old, young}, 1, 2, 0.1);
    const auto& cfg = section.configs.front();
    EXPECT_EQ(cfg.contacts, (std::vector<double>{0, 10}));
    EXPECT_EQ(cfg.T_surf, 282);
    EXPECT_EQ(cfg.layers.front().lithotype, 2);
    EXPECT_DOUBLE_EQ(cfg.layers.front().phi_top, 0.3);
    const auto& epoch = section.epochs.front();
    EXPECT_EQ(epoch.rows, (std::vector<std::size_t>{2, 5}));
    EXPECT_EQ(epoch.born_layer_id, 1);
    EXPECT_DOUBLE_EQ(epoch.faces_at(0)[2], 0.2);
    EXPECT_DOUBLE_EQ(epoch.faces_at(0.5).back(), 12);
    EXPECT_EQ(epoch.faces_at(1), (std::vector<double>{0, 2, 4, 6, 8, 10, 12, 14}));
}
