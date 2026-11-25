#include "basin/physics/thermal_properties.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>

namespace {

constexpr FluidProperties kFluid{
    .lambda = 0.7,
    .rho = 1040.0,
    .c = 4184.0,
};

struct ReferenceCase {
    int code;
    double rho_s;
    double lambda20;
    double c20;
    double A_s;
    double T;
    double phi;
    double lambda_s;
    double c_s;
    double effective_conductivity;
    double rho_c;
    double effective_heat_production;
};

// Conductivities use Sekiguchi (1984) on the mixture; other properties use the organizer data.
constexpr std::array kReferenceCases = {
    ReferenceCase{1001, 2200, 6.5, 862.5, 1.2399999999999999e-08, 273.14999999999998, 0,
                  6.925652336179426, 821.96249999999998, 6.925652336179426, 1808317.5,
                  1.2399999999999999e-08},
    ReferenceCase{1001, 2200, 6.5, 862.5, 1.2399999999999999e-08, 373.14999999999998,
                  0.20000000000000001, 5.253670795900734, 996.05036249999978, 3.541796882454333,
                  2623320.6379999993, 9.9199999999999988e-09},
    ReferenceCase{1001, 2200, 6.5, 862.5, 1.2399999999999999e-08, 473.14999999999998,
                  0.40000000000000002, 4.288434068167193, 1127.3979000000002, 2.2751645807601335,
                  3228709.2280000001, 7.4399999999999991e-09},
    ReferenceCase{1002, 2790, 4.2000000000000002, 862.5, 2.9420000000000003e-07, 273.14999999999998,
                  0, 4.415403995570226, 821.96249999999998, 4.415403995570226, 2293275.375,
                  2.9420000000000003e-07},
    ReferenceCase{1002, 2790, 4.2000000000000002, 862.5, 2.9420000000000003e-07, 373.14999999999998,
                  0.20000000000000001, 3.569287403028196, 996.05036249999978, 2.6429306560700017,
                  3093456.4090999993, 2.3536000000000001e-07},
    ReferenceCase{1002, 2790, 4.2000000000000002, 862.5, 2.9420000000000003e-07, 473.14999999999998,
                  0.40000000000000002, 3.0808233910809157, 1127.3979000000002, 1.9525523343993914,
                  3627808.0846000002, 1.7652e-07},
    ReferenceCase{1003, 2720, 3.9500000000000002, 854.10000000000002, 7.0340000000000001e-07,
                  273.14999999999998, 0, 4.142550915069226, 813.95730000000003, 4.142550915069226,
                  2213963.8560000001, 7.0340000000000001e-07},
    ReferenceCase{1003, 2720, 3.9500000000000002, 854.10000000000002, 7.0340000000000001e-07,
                  373.14999999999998, 0.20000000000000001, 3.3862022516290065, 986.34969809999984,
                  2.539950499708653, 3016568.9430655995, 5.6271999999999995e-07},
    ReferenceCase{1003, 2720, 3.9500000000000002, 854.10000000000002, 7.0340000000000001e-07,
                  473.14999999999998, 0.40000000000000002, 2.949561360962842, 1116.4180248,
                  1.9136194400334061, 3562538.2164735999, 4.2203999999999999e-07},
    ReferenceCase{1004, 2700, 1.6399999999999999, 862.5, 2.0341999999999995e-06, 273.14999999999998,
                  0, 1.6213884512399857, 821.96249999999998, 1.6213884512399857, 2219298.75,
                  2.0341999999999995e-06},
    ReferenceCase{1004, 2700, 1.6399999999999999, 862.5, 2.0341999999999995e-06, 373.14999999999998,
                  0.20000000000000001, 1.6944954527005005, 996.05036249999978, 1.5064530246795784,
                  3021740.7829999994, 1.6273599999999998e-06},
    ReferenceCase{1004, 2700, 1.6399999999999999, 862.5, 2.0341999999999995e-06, 473.14999999999998,
                  0.40000000000000002, 1.736700202671841, 1127.3979000000002, 1.4881756417029717,
                  3566928.5980000002, 1.2205199999999997e-06},
    ReferenceCase{1005, 2720, 2.0499999999999998, 908.5, 9.6069999999999997e-07, 273.14999999999998,
                  0, 2.0688675032616257, 865.80049999999994, 2.0688675032616257, 2354977.3599999999,
                  9.6069999999999997e-07},
    ReferenceCase{1005, 2720, 2.0499999999999998, 908.5, 9.6069999999999997e-07, 373.14999999999998,
                  0.20000000000000001, 1.9947551009951705, 1049.1730484999998, 1.704433628054749,
                  3153272.5535359997, 7.6855999999999999e-07},
    ReferenceCase{1005, 2720, 2.0499999999999998, 908.5, 9.6069999999999997e-07, 473.14999999999998,
                  0.40000000000000002, 1.9519699320654817, 1187.5257880000001, 1.5759317897369698,
                  3678586.0860160002, 5.7641999999999992e-07},
};

void expect_near_reference(const double actual, const double expected) {
    const double tolerance = 5.0e-13 * std::max(1.0, std::abs(expected));
    EXPECT_NEAR(actual, expected, tolerance);
}

LithotypeProperties lithotype_from(const ReferenceCase& reference) {
    return {
        .code = reference.code,
        .rho_s = reference.rho_s,
        .lambda20 = reference.lambda20,
        .c20 = reference.c20,
        .A_s = reference.A_s,
    };
}

} // namespace

TEST(ThermalProperties, MatchesReferenceValuesForAllLithotypes) {
    for (const ReferenceCase& reference : kReferenceCases) {
        const LithotypeProperties lithotype = lithotype_from(reference);
        expect_near_reference(sekiguchi_conductivity(reference.T, lithotype.lambda20),
                              reference.lambda_s);
        expect_near_reference(waples_heat_capacity(reference.T, lithotype.c20), reference.c_s);

        const ThermalProperties effective =
            thermal_properties(reference.T, reference.phi, lithotype, kFluid);
        expect_near_reference(effective.lambda, reference.effective_conductivity);
        expect_near_reference(effective.rho_c, reference.rho_c);
        expect_near_reference(effective.A, reference.effective_heat_production);
    }
}

TEST(ThermalProperties, EvaluatesPreallocatedCellFields) {
    const std::array lithotypes = {lithotype_from(kReferenceCases[0]),
                                   lithotype_from(kReferenceCases[4]),
                                   lithotype_from(kReferenceCases[8])};
    const std::array<const LithotypeProperties*, 3> cell_lithotypes = {
        &lithotypes[0], &lithotypes[1], &lithotypes[2]};
    const std::array T = {kReferenceCases[0].T, kReferenceCases[4].T, kReferenceCases[8].T};
    const std::array phi = {kReferenceCases[0].phi, kReferenceCases[4].phi, kReferenceCases[8].phi};
    std::array<double, 3> lambda{};
    std::array<double, 3> rho_c{};
    std::array<double, 3> A{};

    update_thermal_fields(T, phi, cell_lithotypes, kFluid, lambda, rho_c, A);

    const std::array references = {kReferenceCases[0], kReferenceCases[4], kReferenceCases[8]};
    for (std::size_t index = 0; index < references.size(); ++index) {
        expect_near_reference(lambda[index], references[index].effective_conductivity);
        expect_near_reference(rho_c[index], references[index].rho_c);
        expect_near_reference(A[index], references[index].effective_heat_production);
    }
}
