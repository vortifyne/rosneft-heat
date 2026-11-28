#include "basin_1d/mesh/section_transfer.hpp"

#include <algorithm>
#include <numeric>

SectionLayout::SectionLayout(std::span<const SectionLayer1D> material,
                             std::span<const std::size_t> rows) {
    std::size_t cell = 0;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (!rows[i])
            continue;
        layers.push_back({material[i].id, cell, node_faces.size(), rows[i]});
        for (std::size_t j = 0; j <= rows[i]; ++j)
            node_faces.push_back(cell + j);
        cell += rows[i];
    }
}

namespace {
using Entries = std::vector<Eigen::Triplet<double>>;
void interpolate(Entries& entries, std::size_t row, std::size_t offset, std::span<const double> z,
                 double depth) {
    if (z.size() == 1) {
        entries.emplace_back(row, offset, 1.);
        return;
    }
    const auto upper = std::upper_bound(z.begin() + 1, z.end() - 1, depth);
    const auto j = static_cast<std::size_t>(upper - z.begin());
    const double w = (depth - z[j - 1]) / (z[j] - z[j - 1]);
    entries.emplace_back(row, offset + j - 1, 1. - w);
    entries.emplace_back(row, offset + j, w);
}
} // namespace

SectionTransfer make_section_transfer(const SectionLayout& source, std::span<const double> old_z,
                                      const SectionLayout& target, std::span<const double> new_z,
                                      std::optional<int> new_layer,
                                      std::optional<int> reset_maturity) {
    std::vector<double> centers(old_z.size() - 1);
    for (std::size_t i = 0; i < centers.size(); ++i)
        centers[i] = std::midpoint(old_z[i], old_z[i + 1]);
    SectionTransfer transfer{
        ColTransferOperator(new_z.size() - 1, centers.size()),
        ColTransferOperator(target.node_faces.size(), source.node_faces.size())};
    Entries cells, nodes;
    for (const auto& layer : target.layers) {
        if (layer.id != new_layer)
            for (std::size_t i = layer.cell; i < layer.cell + layer.count; ++i)
                interpolate(cells, i, 0, centers, std::midpoint(new_z[i], new_z[i + 1]));
        if (layer.id == reset_maturity)
            continue;
        const auto old = std::ranges::find(source.layers, layer.id, &SectionLayerRange::id);
        if (old == source.layers.end())
            continue;
        const auto z = old_z.subspan(old->cell, old->count + 1);
        for (std::size_t i = 0; i <= layer.count; ++i) {
            double depth = std::clamp(new_z[layer.cell + i], z.front(), z.back());
            if (i == 0)
                depth = z.front();
            if (i == layer.count)
                depth = z.back();
            interpolate(nodes, layer.node + i, old->node, z, depth);
        }
    }
    transfer.temperature.setFromTriplets(cells.begin(), cells.end());
    transfer.maturity.setFromTriplets(nodes.begin(), nodes.end());
    return transfer;
}
