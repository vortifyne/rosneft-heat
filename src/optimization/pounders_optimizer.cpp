#include "optimization/pounders_optimizer.hpp"

#include <cmath>
#include <exception>
#include <limits>
#include <petsctao.h>
#include <petscvec.h>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

template <typename Handle, PetscErrorCode (*Destroy)(Handle*)> class PetscHandle {
public:
    PetscHandle() = default;

    ~PetscHandle() {
        if (value_ != nullptr) {
            Destroy(&value_);
        }
    }

    PetscHandle(const PetscHandle&) = delete;
    PetscHandle& operator=(const PetscHandle&) = delete;

    Handle get() const noexcept {
        return value_;
    }

    Handle* out() noexcept {
        return &value_;
    }

private:
    Handle value_ = nullptr;
};

using VecHandle = PetscHandle<Vec, VecDestroy>;
using TaoHandle = PetscHandle<Tao, TaoDestroy>;

void check_petsc(PetscErrorCode error, const char* message) {
    if (error != PETSC_SUCCESS) {
        throw std::runtime_error(std::string(message) + " (PETSc error " + std::to_string(error) +
                                 ")");
    }
}

PetscInt to_petsc_size(Vector::Index size, const char* name) {
    if (size <= 0 || size > static_cast<Vector::Index>(std::numeric_limits<PetscInt>::max())) {
        throw std::invalid_argument(std::string(name) + " must fit into PetscInt and be positive");
    }
    return static_cast<PetscInt>(size);
}

void validate_request(const PoundersSolveRequest& request) {
    if (!std::isfinite(request.gradient_absolute_tolerance) ||
        request.gradient_absolute_tolerance < 0.0) {
        throw std::invalid_argument("Absolute gradient tolerance must be finite and non-negative");
    }
    if (!std::isfinite(request.gradient_relative_tolerance) ||
        request.gradient_relative_tolerance < 0.0) {
        throw std::invalid_argument("Relative gradient tolerance must be finite and non-negative");
    }
    if (!std::isfinite(request.gradient_reduction_tolerance) ||
        request.gradient_reduction_tolerance < 0.0) {
        throw std::invalid_argument("Gradient reduction tolerance must be finite and non-negative");
    }
    if (request.max_iterations <= 0) {
        throw std::invalid_argument("Maximum iteration count must be positive");
    }
    if (request.max_function_evaluations <= 0) {
        throw std::invalid_argument("Maximum function evaluation count must be positive");
    }
}

void validate_problem(const BoundedLeastSquaresProblem& problem, const Vector& initial_parameters) {
    if (!problem.residual_function) {
        throw std::invalid_argument("Least-squares residual function must be set");
    }
    if (problem.lower_bounds.empty()) {
        throw std::invalid_argument("Least-squares problem must have at least one parameter");
    }
    if (problem.lower_bounds.size() != problem.upper_bounds.size() ||
        problem.lower_bounds.size() != initial_parameters.size()) {
        throw std::invalid_argument("Parameter vectors and bound vectors must have the same size");
    }
    if (!problem.lower_bounds.all_finite() || !problem.upper_bounds.all_finite() ||
        !initial_parameters.all_finite()) {
        throw std::invalid_argument("Parameters and bounds must be finite");
    }
    if (problem.residual_count <= 0) {
        throw std::invalid_argument("Least-squares problem must have at least one residual");
    }

    for (Vector::Index index = 0; index < initial_parameters.size(); ++index) {
        if (!(problem.lower_bounds[index] < problem.upper_bounds[index])) {
            throw std::invalid_argument("Parameter bounds must be strictly increasing");
        }
        if (initial_parameters[index] < problem.lower_bounds[index] ||
            initial_parameters[index] > problem.upper_bounds[index]) {
            throw std::invalid_argument("Initial parameters must lie within their bounds");
        }
    }

    static_cast<void>(to_petsc_size(problem.lower_bounds.size(), "Parameter count"));
    static_cast<void>(to_petsc_size(problem.residual_count, "Residual count"));
}

void ensure_petsc_is_active() {
    PetscBool initialized = PETSC_FALSE;
    PetscBool finalized = PETSC_FALSE;
    check_petsc(PetscInitialized(&initialized), "Failed to query PETSc initialization state");
    check_petsc(PetscFinalized(&finalized), "Failed to query PETSc finalization state");
    if (initialized == PETSC_FALSE || finalized == PETSC_TRUE) {
        throw std::logic_error(
            "PoundersOptimizer requires PETSc to be initialized and not finalized");
    }
}

void copy_to_petsc(const Vector& source, Vec destination) {
    PetscScalar* values = nullptr;
    check_petsc(VecGetArray(destination, &values), "Failed to access writable PETSc vector data");
    for (Vector::Index index = 0; index < source.size(); ++index) {
        values[index] = static_cast<PetscScalar>(source[index]);
    }
    check_petsc(VecRestoreArray(destination, &values), "Failed to restore PETSc vector data");
}

Vector copy_from_petsc(Vec source, Vector::Index size) {
    const PetscScalar* values = nullptr;
    check_petsc(VecGetArrayRead(source, &values), "Failed to access readable PETSc vector data");
    Vector result(size);
    for (Vector::Index index = 0; index < size; ++index) {
        result[index] = static_cast<double>(PetscRealPart(values[index]));
    }
    check_petsc(VecRestoreArrayRead(source, &values), "Failed to restore PETSc vector data");
    return result;
}

struct ResidualCallbackContext {
    const BoundedLeastSquaresProblem* problem;
    std::exception_ptr exception;
};

PetscErrorCode evaluate_residual(Tao tao, Vec parameters, Vec residuals, void* raw_context) {
    static_cast<void>(tao);
    auto* context = static_cast<ResidualCallbackContext*>(raw_context);
    if (context->exception != nullptr) {
        return PETSC_ERR_USER;
    }

    try {
        const Vector parameter_values =
            copy_from_petsc(parameters, context->problem->lower_bounds.size());
        const Vector residual_values = context->problem->residual_function(parameter_values);
        if (residual_values.size() != context->problem->residual_count) {
            throw std::runtime_error("Residual function returned a vector with an invalid size");
        }
        if (!residual_values.all_finite()) {
            throw std::runtime_error("Residual function returned non-finite values");
        }
        copy_to_petsc(residual_values, residuals);
    } catch (...) {
        context->exception = std::current_exception();
        return PETSC_ERR_USER;
    }
    return PETSC_SUCCESS;
}

PoundersSolveStatus to_status(TaoConvergedReason reason) {
    switch (reason) {
    case TAO_CONVERGED_GATOL:
        return PoundersSolveStatus::converged_gradient_absolute;
    case TAO_CONVERGED_GRTOL:
        return PoundersSolveStatus::converged_gradient_relative;
    case TAO_CONVERGED_GTTOL:
        return PoundersSolveStatus::converged_gradient_reduction;
    case TAO_CONVERGED_STEPTOL:
        return PoundersSolveStatus::converged_step;
    case TAO_CONVERGED_MINF:
        return PoundersSolveStatus::converged_objective;
    case TAO_CONVERGED_USER:
        return PoundersSolveStatus::converged_user;
    case TAO_DIVERGED_MAXITS:
        return PoundersSolveStatus::maximum_iterations;
    case TAO_DIVERGED_MAXFCN:
        return PoundersSolveStatus::maximum_function_evaluations;
    case TAO_DIVERGED_NAN:
        return PoundersSolveStatus::nonfinite_objective;
    case TAO_DIVERGED_LS_FAILURE:
        return PoundersSolveStatus::line_search_failed;
    case TAO_DIVERGED_TR_REDUCTION:
        return PoundersSolveStatus::trust_region_failed;
    case TAO_DIVERGED_USER:
        return PoundersSolveStatus::diverged_user;
    case TAO_CONTINUE_ITERATING:
    default:
        return PoundersSolveStatus::unknown;
    }
}

int to_int(PetscInt value, const char* name) {
    if (value < static_cast<PetscInt>(std::numeric_limits<int>::min()) ||
        value > static_cast<PetscInt>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(std::string(name) + " does not fit into int");
    }
    return static_cast<int>(value);
}

} // namespace

bool PoundersSolveResult::converged() const noexcept {
    switch (status) {
    case PoundersSolveStatus::converged_gradient_absolute:
    case PoundersSolveStatus::converged_gradient_relative:
    case PoundersSolveStatus::converged_gradient_reduction:
    case PoundersSolveStatus::converged_step:
    case PoundersSolveStatus::converged_objective:
    case PoundersSolveStatus::converged_user:
        return true;
    default:
        return false;
    }
}

PoundersSolveResult PoundersOptimizer::solve(const BoundedLeastSquaresProblem& problem,
                                             const Vector& initial_parameters,
                                             const PoundersSolveRequest& request) const {
    validate_problem(problem, initial_parameters);
    validate_request(request);
    ensure_petsc_is_active();

    const PetscInt parameter_count = to_petsc_size(initial_parameters.size(), "Parameter count");
    const PetscInt residual_count = to_petsc_size(problem.residual_count, "Residual count");

    VecHandle parameters;
    VecHandle residuals;
    VecHandle lower_bounds;
    VecHandle upper_bounds;
    check_petsc(VecCreateSeq(PETSC_COMM_SELF, parameter_count, parameters.out()),
                "Failed to create sequential POUNDERS parameter vector");
    check_petsc(VecCreateSeq(PETSC_COMM_SELF, residual_count, residuals.out()),
                "Failed to create sequential POUNDERS residual vector");
    check_petsc(VecCreateSeq(PETSC_COMM_SELF, parameter_count, lower_bounds.out()),
                "Failed to create sequential POUNDERS lower-bound vector");
    check_petsc(VecCreateSeq(PETSC_COMM_SELF, parameter_count, upper_bounds.out()),
                "Failed to create sequential POUNDERS upper-bound vector");
    copy_to_petsc(initial_parameters, parameters.get());
    copy_to_petsc(problem.lower_bounds, lower_bounds.get());
    copy_to_petsc(problem.upper_bounds, upper_bounds.get());

    ResidualCallbackContext callback_context{.problem = &problem, .exception = nullptr};
    TaoHandle tao;
    check_petsc(TaoCreate(PETSC_COMM_SELF, tao.out()), "Failed to create PETSc TAO solver");
    check_petsc(TaoSetType(tao.get(), TAOPOUNDERS), "Failed to select PETSc POUNDERS");
    check_petsc(TaoSetSolution(tao.get(), parameters.get()),
                "Failed to set POUNDERS initial parameters");
    check_petsc(
        TaoSetResidualRoutine(tao.get(), residuals.get(), evaluate_residual, &callback_context),
        "Failed to set POUNDERS residual function");
    check_petsc(TaoSetVariableBounds(tao.get(), lower_bounds.get(), upper_bounds.get()),
                "Failed to set POUNDERS parameter bounds");
    check_petsc(TaoSetTolerances(tao.get(), request.gradient_absolute_tolerance,
                                 request.gradient_relative_tolerance,
                                 request.gradient_reduction_tolerance),
                "Failed to set POUNDERS tolerances");
    check_petsc(TaoSetMaximumIterations(tao.get(), request.max_iterations),
                "Failed to set POUNDERS iteration limit");
    check_petsc(TaoSetMaximumFunctionEvaluations(tao.get(), request.max_function_evaluations),
                "Failed to set POUNDERS function evaluation limit");
    check_petsc(TaoSetFromOptions(tao.get()), "Failed to apply PETSc TAO options");

    const PetscErrorCode solve_error = TaoSolve(tao.get());
    if (callback_context.exception != nullptr) {
        std::rethrow_exception(callback_context.exception);
    }
    check_petsc(solve_error, "PETSc POUNDERS solve failed");

    TaoConvergedReason reason = TAO_CONTINUE_ITERATING;
    PetscInt iterations = 0;
    PetscInt function_evaluations = 0;
    check_petsc(TaoGetConvergedReason(tao.get(), &reason),
                "Failed to get POUNDERS convergence reason");
    check_petsc(TaoGetIterationNumber(tao.get(), &iterations),
                "Failed to get POUNDERS iteration count");
    check_petsc(TaoGetCurrentFunctionEvaluations(tao.get(), &function_evaluations),
                "Failed to get POUNDERS function evaluation count");

    Vector solution = copy_from_petsc(parameters.get(), initial_parameters.size());
    Vector final_residuals = copy_from_petsc(residuals.get(), problem.residual_count);
    const double objective = final_residuals.squared_norm();
    return {
        .status = to_status(reason),
        .parameters = std::move(solution),
        .residuals = std::move(final_residuals),
        .objective = objective,
        .iterations = to_int(iterations, "POUNDERS iteration count"),
        .function_evaluations = to_int(function_evaluations, "POUNDERS function evaluation count"),
    };
}
