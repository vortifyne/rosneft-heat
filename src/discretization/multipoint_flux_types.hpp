#pragma once

#include <cstddef>
#include <span>
#include <vector>

enum class MultipointFluxScheme {
    linear,
    // Diagnostic baseline retained to isolate the historical 1/2--1/2
    // combination from conductance-weighted linear fluxes on C grids.
    linear_equal_weight,
    nonlinear_conservative,
    nonlinear_dmp,
    nonlinear_two_point,
};

struct MultipointFluxTerm {
    std::size_t cell = 0;
    double coefficient = 0.0;
};

struct MultipointFluxExpression {
    std::vector<MultipointFluxTerm> terms;
    double constant = 0.0;

    [[nodiscard]] double value(std::span<const double> solution) const;
};
