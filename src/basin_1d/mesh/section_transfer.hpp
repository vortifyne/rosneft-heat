#pragma once

#include "basin/mesh/col_field_transfer.hpp"
#include "basin_1d/geometry/basin_section_1d.hpp"

struct SectionLayerRange {
    int id;
    std::size_t cell, node, count;
};

struct SectionLayout {
    std::vector<SectionLayerRange> layers;
    std::vector<std::size_t>
        node_faces; // Adjacent layers have separate maturity nodes at a contact.
    SectionLayout(std::span<const SectionLayer1D> layers, std::span<const std::size_t> rows);
};

struct SectionTransfer {
    ColTransferOperator temperature, maturity;
};

SectionTransfer make_section_transfer(const SectionLayout& source, std::span<const double> old_z,
                                      const SectionLayout& target, std::span<const double> new_z,
                                      std::optional<int> new_layer,
                                      std::optional<int> reset_maturity);
