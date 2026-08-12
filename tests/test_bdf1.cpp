#include "time/bdf1.hpp"

#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

namespace {

void expect_vector_equal(const Vector& actual, const Vector& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (Vector::Index index = 0; index < actual.size(); ++index) {
        EXPECT_DOUBLE_EQ(actual[index], expected[index]);
    }
}

class RecordingSemiDiscreteSystem final : public SemiDiscreteSystem {
public:
    void assemble_residual(double time, const Vector& solution, const Vector& solution_derivative,
                           Vector& residual) const override {
        residual_calls++;
        last_time = time;
        last_solution = solution;
        last_solution_derivative = solution_derivative;
        residual = solution_derivative;
    }

    void assemble_matrix(NonlinearMethod method, double time, const Vector& solution,
                         const Vector& solution_derivative, double derivative_shift,
                         SparseMatrix& matrix) const override {
        matrix_calls++;
        last_method = method;
        last_time = time;
        last_solution = solution;
        last_solution_derivative = solution_derivative;
        last_derivative_shift = derivative_shift;
        matrix.set_zero();
    }

    mutable int residual_calls = 0;
    mutable int matrix_calls = 0;
    mutable NonlinearMethod last_method = NonlinearMethod::picard;
    mutable double last_time = 0.0;
    mutable double last_derivative_shift = 0.0;
    mutable Vector last_solution;
    mutable Vector last_solution_derivative;
};

TimeHistory make_history() {
    TimeHistory history;
    history.accept({.time = 2.0, .solution = Vector{1.0, 4.0}});
    return history;
}

TEST(BDF1Test, RequiresOnePreviousSnapshot) {
    const BDF1 scheme;

    EXPECT_EQ(scheme.required_snapshot_count(), 1);
}

TEST(BDF1Test, AssemblesResidualAtNewTimeWithBdf1Derivative) {
    const BDF1 scheme;
    const TimeHistory history = make_history();
    RecordingSemiDiscreteSystem semi_discrete_system;
    auto nonlinear_system = scheme.make_nonlinear_system(semi_discrete_system, history, 2.5);
    const Vector solution{2.0, 7.0};
    Vector residual;

    nonlinear_system->assemble_residual(solution, residual);

    EXPECT_EQ(semi_discrete_system.residual_calls, 1);
    EXPECT_DOUBLE_EQ(semi_discrete_system.last_time, 2.5);
    expect_vector_equal(semi_discrete_system.last_solution, solution);
    expect_vector_equal(semi_discrete_system.last_solution_derivative, Vector{2.0, 6.0});
    expect_vector_equal(residual, Vector{2.0, 6.0});
}

TEST(BDF1Test, AssemblesPicardMatrixWithBdf1DerivativeShift) {
    const BDF1 scheme;
    const TimeHistory history = make_history();
    RecordingSemiDiscreteSystem semi_discrete_system;
    auto nonlinear_system = scheme.make_nonlinear_system(semi_discrete_system, history, 2.5);
    const Vector solution{2.0, 7.0};
    SparseMatrix matrix(2, 2, SparseStorageOrder::csc);

    nonlinear_system->assemble_matrix(NonlinearMethod::picard, solution, matrix);

    EXPECT_EQ(semi_discrete_system.matrix_calls, 1);
    EXPECT_EQ(semi_discrete_system.last_method, NonlinearMethod::picard);
    EXPECT_DOUBLE_EQ(semi_discrete_system.last_time, 2.5);
    EXPECT_DOUBLE_EQ(semi_discrete_system.last_derivative_shift, 2.0);
    expect_vector_equal(semi_discrete_system.last_solution, solution);
    expect_vector_equal(semi_discrete_system.last_solution_derivative, Vector{2.0, 6.0});
}

TEST(BDF1Test, AssemblesNewtonMatrixWithBdf1DerivativeShift) {
    const BDF1 scheme;
    const TimeHistory history = make_history();
    RecordingSemiDiscreteSystem semi_discrete_system;
    auto nonlinear_system = scheme.make_nonlinear_system(semi_discrete_system, history, 2.25);
    const Vector solution{2.0, 7.0};
    SparseMatrix matrix(2, 2, SparseStorageOrder::csc);

    nonlinear_system->assemble_matrix(NonlinearMethod::newton, solution, matrix);

    EXPECT_EQ(semi_discrete_system.matrix_calls, 1);
    EXPECT_EQ(semi_discrete_system.last_method, NonlinearMethod::newton);
    EXPECT_DOUBLE_EQ(semi_discrete_system.last_time, 2.25);
    EXPECT_DOUBLE_EQ(semi_discrete_system.last_derivative_shift, 4.0);
    expect_vector_equal(semi_discrete_system.last_solution_derivative, Vector{4.0, 12.0});
}

TEST(BDF1Test, RejectsInvalidTimeSteps) {
    const BDF1 scheme;
    const TimeHistory history = make_history();
    const RecordingSemiDiscreteSystem semi_discrete_system;

    EXPECT_THROW(scheme.make_nonlinear_system(semi_discrete_system, history, 2.0),
                 std::invalid_argument);
    EXPECT_THROW(scheme.make_nonlinear_system(semi_discrete_system, history, 1.0),
                 std::invalid_argument);
    EXPECT_THROW(scheme.make_nonlinear_system(semi_discrete_system, history,
                                              std::numeric_limits<double>::quiet_NaN()),
                 std::invalid_argument);
    EXPECT_THROW(scheme.make_nonlinear_system(semi_discrete_system, history,
                                              std::numeric_limits<double>::infinity()),
                 std::invalid_argument);
}

TEST(BDF1Test, RejectsEmptyHistory) {
    const BDF1 scheme;
    const TimeHistory history;
    const RecordingSemiDiscreteSystem semi_discrete_system;

    EXPECT_THROW(scheme.make_nonlinear_system(semi_discrete_system, history, 1.0),
                 std::logic_error);
}

TEST(BDF1Test, RejectsNonfinitePreviousOrStepEndTime) {
    const BDF1 scheme;
    const RecordingSemiDiscreteSystem semi_discrete_system;
    TimeHistory nonfinite_history;
    nonfinite_history.accept(
        {.time = std::numeric_limits<double>::infinity(), .solution = Vector{1.0}});

    EXPECT_THROW(scheme.make_nonlinear_system(semi_discrete_system, nonfinite_history, 1.0),
                 std::logic_error);
    EXPECT_THROW(scheme.make_nonlinear_system(semi_discrete_system, make_history(),
                                              std::numeric_limits<double>::infinity()),
                 std::invalid_argument);
}

TEST(BDF1Test, RejectsSolutionWithDifferentSizeFromHistory) {
    const BDF1 scheme;
    const TimeHistory history = make_history();
    RecordingSemiDiscreteSystem semi_discrete_system;
    auto nonlinear_system = scheme.make_nonlinear_system(semi_discrete_system, history, 2.5);
    const Vector solution{2.0};
    Vector residual;

    EXPECT_THROW(nonlinear_system->assemble_residual(solution, residual), std::invalid_argument);
}

} // namespace
