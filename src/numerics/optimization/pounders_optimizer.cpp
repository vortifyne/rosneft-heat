#include "numerics/optimization/pounders_optimizer.hpp"

#include <cmath>
#include <exception>
#include <limits>
#include <numbers>
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
    const LeastSquaresObserver* observer;
    std::exception_ptr exception;
    int evaluations = 0;
    double residual_tolerance = 0.0;
    double objective = std::numeric_limits<double>::infinity();
    Vector parameters{};
    Vector residuals{};
    bool within_errors = false;
};

PetscErrorCode evaluate_residual(Tao tao, Vec px, Vec pr, void* raw_context) {
    auto* context = static_cast<ResidualCallbackContext*>(raw_context);
    if (context->exception != nullptr) {
        return PETSC_ERR_USER;
    }

    try {
        Vector x = copy_from_petsc(px, context->problem->lower_bounds.size());
        x.native() = x.native()
                         .cwiseMax(context->problem->lower_bounds.native())
                         .cwiseMin(context->problem->upper_bounds.native());
        auto [r, within_errors] = context->problem->evaluate(x, context->residual_tolerance);
        if (r.size() != context->problem->residual_count) {
            throw std::runtime_error("Residual function returned a vector with an invalid size");
        }
        if (!r.all_finite()) {
            throw std::runtime_error("Residual function returned non-finite values");
        }
        copy_to_petsc(r, pr);
        // TAO POUNDERS sums squared residuals, so scale its private copy to obtain half the sum.
        check_petsc(VecScale(pr, 1.0 / std::numbers::sqrt2), "Failed to scale POUNDERS residuals");
        ++context->evaluations;
        if (*context->observer) {
            PetscInt iteration = 0;
            check_petsc(TaoGetIterationNumber(tao, &iteration), "Failed to get POUNDERS iteration");
            (*context->observer)(static_cast<int>(iteration), context->evaluations, x, r);
        }
        const double objective = least_squares_objective(r);
        if ((within_errors && !context->within_errors) ||
            (within_errors == context->within_errors && objective < context->objective)) {
            context->parameters = std::move(x);
            context->residuals = std::move(r);
            context->objective = objective;
            context->within_errors = within_errors;
        }
    } catch (...) {
        context->exception = std::current_exception();
        return PETSC_ERR_USER;
    }
    return PETSC_SUCCESS;
}

PetscErrorCode check_residual_tolerance(Tao tao, void* raw_context) {
    const auto& context = *static_cast<ResidualCallbackContext*>(raw_context);
    PetscFunctionBeginUser;
    PetscCall(TaoDefaultConvergenceTest(tao, nullptr));
    if (context.within_errors)
        PetscCall(TaoSetConvergedReason(tao, TAO_CONVERGED_USER));
    PetscFunctionReturn(PETSC_SUCCESS);
}

LeastSquaresStatus to_status(TaoConvergedReason reason) {
    if (reason == TAO_CONVERGED_USER)
        return LeastSquaresStatus::within_errors;
    if (reason == TAO_DIVERGED_MAXITS || reason == TAO_DIVERGED_MAXFCN)
        return LeastSquaresStatus::limit_reached;
    return reason > 0 ? LeastSquaresStatus::converged : LeastSquaresStatus::failed;
}

int to_int(PetscInt value, const char* name) {
    if (value < static_cast<PetscInt>(std::numeric_limits<int>::min()) ||
        value > static_cast<PetscInt>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(std::string(name) + " does not fit into int");
    }
    return static_cast<int>(value);
}

} // namespace

LeastSquaresSolveResult PoundersOptimizer::solve(const BoundedLeastSquaresProblem& problem,
                                                 const Vector& x0,
                                                 const LeastSquaresSolveRequest& request,
                                                 const LeastSquaresObserver& observe) const {
    if (!problem.residual_function && !problem.objective)
        throw std::invalid_argument("Least-squares residual function must be set");
    problem.validate(x0);
    request.validate();
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
    PetscReal delta = 0.1;
    check_petsc(PetscOptionsGetReal(nullptr, nullptr, "-tao_pounders_delta", &delta, nullptr),
                "Failed to read POUNDERS initial step");
    Vector initial = x0;
    initial.native() =
        initial.native().cwiseMin((problem.upper_bounds.native().array() - delta).matrix());
    copy_to_petsc(initial, px.get());
    copy_to_petsc(problem.lower_bounds, lower.get());
    copy_to_petsc(problem.upper_bounds, upper.get());

    ResidualCallbackContext context{.problem = &problem,
                                    .observer = &observe,
                                    .exception = nullptr,
                                    .residual_tolerance = request.residual_tolerance};
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
    if (request.residual_tolerance > 0.0)
        check_petsc(TaoSetConvergenceTest(tao.get(), check_residual_tolerance, &context),
                    "Failed to set residual convergence test");

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

    return {
        .status = to_status(reason),
        .parameters = std::move(context.parameters),
        .residuals = std::move(context.residuals),
        .objective = context.objective,
        .iterations = to_int(iterations, "POUNDERS iteration count"),
        .function_evaluations = to_int(function_evaluations, "POUNDERS function evaluation count"),
    };
}
