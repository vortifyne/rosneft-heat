#pragma once

#include <cstddef>
#include <span>
#include <vector>

enum class MultipointFluxScheme {
    linear,
    nonlinear_conservative,
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
