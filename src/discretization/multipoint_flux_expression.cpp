#include "discretization/multipoint_flux.hpp"

double MultipointFluxExpression::value(const std::span<const double> solution) const {
    double result = constant;
    for (const MultipointFluxTerm term : terms) {
        result += term.coefficient * solution[term.cell];
    }
    return result;
}
