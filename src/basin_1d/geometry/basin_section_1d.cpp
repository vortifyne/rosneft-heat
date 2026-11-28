#include "basin_1d/geometry/basin_section_1d.hpp"

#include "basin/mesh/vertical_grid.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

std::vector<double> SectionEpoch1D::faces_at(double theta) const {
    if (!(theta >= 0.0 && theta <= 1.0))
        throw std::invalid_argument("Invalid epoch fraction");
    return vertical_epoch_faces(start_contacts, end_contacts, rows, born_layer_id.has_value(),
                                theta, dz_min);
}

BasinSection1D make_basin_section_1d(std::span<const BasinConfig> configs, double x, double dz,
                                     double dz_min) {
    if (configs.size() < 2 || !std::isfinite(dz) || dz <= 0 || !std::isfinite(dz_min) ||
        dz_min <= 0)
        throw std::invalid_argument(
            "Section requires two or more configurations and positive vertical steps");

    BasinSection1D result;
    for (const auto& cfg : configs) {
        const auto present = [x](const BasinLayerProfile& layer) {
            return !layer.x.empty() && layer.x.front() <= x && x <= layer.x.back() &&
                   layer.bottom_at(x) > layer.top_at(x);
        };
        const auto first = std::find_if(cfg.layers.begin(), cfg.layers.end(), present);
        if (first == cfg.layers.end())
            throw std::invalid_argument("Section coordinate has no basin material");
        SectionConfig1D section{cfg.age_ma, cfg.T_surf.at(x), {}, {first->top_at(x)}};
        for (const auto& layer : cfg.layers) {
            double bottom = section.contacts.back();
            if (present(layer)) {
                if (std::abs(layer.top_at(x) - bottom) > 1e-7)
                    throw std::invalid_argument(
                        "Layer boundaries do not form a continuous section");
                bottom = layer.bottom_at(x);
            }
            section.contacts.push_back(bottom);
            section.layers.push_back(
                {layer.id, layer.lithotype_at(x), layer.phi_at(x, 0.0), layer.phi_at(x, 1.0)});
        }

        if (!result.configs.empty()) {
            const auto& old = result.configs.back();
            const bool birth = section.layers.size() == old.layers.size() + 1;
            if (!(old.age_ma > section.age_ma) ||
                (!birth && section.layers.size() != old.layers.size()))
                throw std::invalid_argument("Invalid epoch in basin section");
            for (std::size_t i = 0; i < old.layers.size(); ++i)
                if (old.layers[i].id != section.layers[i + birth].id)
                    throw std::invalid_argument("Layer order changed across section epoch");
            SectionEpoch1D epoch{old.contacts, section.contacts, {}, std::nullopt, dz_min};
            if (birth) {
                epoch.born_layer_id = section.layers.front().id;
                epoch.start_contacts.insert(epoch.start_contacts.begin(), old.contacts.front());
            }
            epoch.rows = vertical_epoch_rows(epoch.start_contacts, epoch.end_contacts, dz);
            result.epochs.push_back(std::move(epoch));
        }
        result.configs.push_back(std::move(section));
    }
    return result;
}
