#include "numerics/optimization/pounders_optimizer.hpp"

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
    if (!std::isfinite(request.gatol) || request.gatol < 0.0) {
        throw std::invalid_argument("Absolute gradient tolerance must be finite and non-negative");
    }
    if (!std::isfinite(request.grtol) || request.grtol < 0.0) {
        throw std::invalid_argument("Relative gradient tolerance must be finite and non-negative");
    }
    if (!std::isfinite(request.gttol) || request.gttol < 0.0) {
        throw std::invalid_argument("Gradient reduction tolerance must be finite and non-negative");
    }
    if (request.maxiter <= 0) {
        throw std::invalid_argument("Maximum iteration count must be positive");
    }
    if (request.maxfev <= 0) {
        throw std::invalid_argument("Maximum function evaluation count must be positive");
    }
}

void validate_problem(const BoundedLeastSquaresProblem& problem, const Vector& x0) {
    if (!problem.residual_function) {
        throw std::invalid_argument("Least-squares residual function must be set");
    }
    if (problem.lower_bounds.empty()) {
        throw std::invalid_argument("Least-squares problem must have at least one parameter");
    }
    if (problem.lower_bounds.size() != problem.upper_bounds.size() ||
        problem.lower_bounds.size() != x0.size()) {
        throw std::invalid_argument("Parameter vectors and bound vectors must have the same size");
    }
    if (!problem.lower_bounds.all_finite() || !problem.upper_bounds.all_finite() ||
        !x0.all_finite()) {
        throw std::invalid_argument("Parameters and bounds must be finite");
    }
    if (problem.residual_count <= 0) {
        throw std::invalid_argument("Least-squares problem must have at least one residual");
    }

    for (Vector::Index index = 0; index < x0.size(); ++index) {
        if (!(problem.lower_bounds[index] < problem.upper_bounds[index])) {
            throw std::invalid_argument("Parameter bounds must be strictly increasing");
        }
        if (x0[index] < problem.lower_bounds[index] || x0[index] > problem.upper_bounds[index]) {
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

PetscErrorCode evaluate_residual(Tao tao, Vec px, Vec pr, void* raw_context) {
    static_cast<void>(tao);
    auto* context = static_cast<ResidualCallbackContext*>(raw_context);
    if (context->exception != nullptr) {
        return PETSC_ERR_USER;
    }

    try {
        const Vector x = copy_from_petsc(px, context->problem->lower_bounds.size());
        const Vector r = context->problem->residual_function(x);
        if (r.size() != context->problem->residual_count) {
            throw std::runtime_error("Residual function returned a vector with an invalid size");
        }
        if (!r.all_finite()) {
            throw std::runtime_error("Residual function returned non-finite values");
        }
        copy_to_petsc(r, pr);
    } catch (...) {
        context->exception = std::current_exception();
        return PETSC_ERR_USER;
    }
    return PETSC_SUCCESS;
}

PoundersStatus to_status(TaoConvergedReason reason) {
    switch (reason) {
    case TAO_CONVERGED_GATOL:
        return PoundersStatus::gatol;
    case TAO_CONVERGED_GRTOL:
        return PoundersStatus::grtol;
    case TAO_CONVERGED_GTTOL:
        return PoundersStatus::gttol;
    case TAO_CONVERGED_STEPTOL:
        return PoundersStatus::step_tol;
    case TAO_CONVERGED_MINF:
        return PoundersStatus::min_objective;
    case TAO_CONVERGED_USER:
        return PoundersStatus::user_converged;
    case TAO_DIVERGED_MAXITS:
        return PoundersStatus::maxiter;
    case TAO_DIVERGED_MAXFCN:
        return PoundersStatus::maxfev;
    case TAO_DIVERGED_NAN:
        return PoundersStatus::nonfinite;
    case TAO_DIVERGED_LS_FAILURE:
        return PoundersStatus::ls_failed;
    case TAO_DIVERGED_TR_REDUCTION:
        return PoundersStatus::tr_failed;
    case TAO_DIVERGED_USER:
        return PoundersStatus::user_diverged;
    case TAO_CONTINUE_ITERATING:
    default:
        return PoundersStatus::unknown;
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

PoundersSolveResult PoundersOptimizer::solve(const BoundedLeastSquaresProblem& problem,
                                             const Vector& x0,
                                             const PoundersSolveRequest& request) const {
    validate_problem(problem, x0);
    validate_request(request);
    ensure_petsc_is_active();

    const PetscInt n = to_petsc_size(x0.size(), "Parameter count");
    const PetscInt m = to_petsc_size(problem.residual_count, "Residual count");

    VecHandle px;
    VecHandle pr;
    VecHandle lower;
    VecHandle upper;
    check_petsc(VecCreateSeq(PETSC_COMM_SELF, n, px.out()),
                "Failed to create sequential POUNDERS parameter vector");
    check_petsc(VecCreateSeq(PETSC_COMM_SELF, m, pr.out()),
                "Failed to create sequential POUNDERS residual vector");
    check_petsc(VecCreateSeq(PETSC_COMM_SELF, n, lower.out()),
                "Failed to create sequential POUNDERS lower-bound vector");
    check_petsc(VecCreateSeq(PETSC_COMM_SELF, n, upper.out()),
                "Failed to create sequential POUNDERS upper-bound vector");
    copy_to_petsc(x0, px.get());
    copy_to_petsc(problem.lower_bounds, lower.get());
    copy_to_petsc(problem.upper_bounds, upper.get());

    ResidualCallbackContext context{.problem = &problem, .exception = nullptr};
    TaoHandle tao;
    check_petsc(TaoCreate(PETSC_COMM_SELF, tao.out()), "Failed to create PETSc TAO solver");
    check_petsc(TaoSetType(tao.get(), TAOPOUNDERS), "Failed to select PETSc POUNDERS");
    check_petsc(TaoSetSolution(tao.get(), px.get()), "Failed to set POUNDERS initial parameters");
    check_petsc(TaoSetResidualRoutine(tao.get(), pr.get(), evaluate_residual, &context),
                "Failed to set POUNDERS residual function");
    check_petsc(TaoSetVariableBounds(tao.get(), lower.get(), upper.get()),
                "Failed to set POUNDERS parameter bounds");
    check_petsc(TaoSetTolerances(tao.get(), request.gatol, request.grtol, request.gttol),
                "Failed to set POUNDERS tolerances");
    check_petsc(TaoSetMaximumIterations(tao.get(), request.maxiter),
                "Failed to set POUNDERS iteration limit");
    check_petsc(TaoSetMaximumFunctionEvaluations(tao.get(), request.maxfev),
                "Failed to set POUNDERS function evaluation limit");
    check_petsc(TaoSetFromOptions(tao.get()), "Failed to apply PETSc TAO options");

    const PetscErrorCode solve_error = TaoSolve(tao.get());
    if (context.exception != nullptr) {
        std::rethrow_exception(context.exception);
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

    Vector x = copy_from_petsc(px.get(), x0.size());
    Vector r_final = copy_from_petsc(pr.get(), problem.residual_count);
    const double objective = r_final.squared_norm();
    return {
        .status = to_status(reason),
        .parameters = std::move(x),
        .residuals = std::move(r_final),
        .objective = objective,
        .iterations = to_int(iterations, "POUNDERS iteration count"),
        .function_evaluations = to_int(function_evaluations, "POUNDERS function evaluation count"),
    };
}
