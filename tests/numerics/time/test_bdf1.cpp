#include "numerics/time/bdf1.hpp"

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
    void assemble_residual(double t, const Vector& u, const Vector& du_dt,
                           Vector& r) const override {
        residual_calls++;
        last_time = t;
        last_solution = u;
        last_solution_derivative = du_dt;
        r = du_dt;
    }

    void assemble_matrix(NonlinearMethod method, double t, const Vector& u, const Vector& du_dt,
                         double alpha, SparseMatrix& A) const override {
        matrix_calls++;
        last_method = method;
        last_time = t;
        last_solution = u;
        last_solution_derivative = du_dt;
        last_derivative_shift = alpha;
        A.set_zero();
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
    history.accept({.t = 2.0, .u = Vector{1.0, 4.0}});
    return history;
}

TEST(BDF1Test, AssemblesResidualAtNewTimeWithBdf1Derivative) {
    const BDF1 scheme;
    const TimeHistory history = make_history();
    RecordingSemiDiscreteSystem system;
    auto nonlinear_system = scheme.make_nonlinear_system(system, history, 2.5);
    const Vector u{2.0, 7.0};
    Vector r;

    nonlinear_system->assemble_residual(u, r);

    EXPECT_EQ(system.residual_calls, 1);
    EXPECT_DOUBLE_EQ(system.last_time, 2.5);
    expect_vector_equal(system.last_solution, u);
    expect_vector_equal(system.last_solution_derivative, Vector{2.0, 6.0});
    expect_vector_equal(r, Vector{2.0, 6.0});
}

TEST(BDF1Test, AssemblesPicardMatrixWithBdf1DerivativeShift) {
    const BDF1 scheme;
    const TimeHistory history = make_history();
    RecordingSemiDiscreteSystem system;
    auto nonlinear_system = scheme.make_nonlinear_system(system, history, 2.5);
    const Vector u{2.0, 7.0};
    SparseMatrix A(2, 2, SparseStorageOrder::csc);

    nonlinear_system->assemble_matrix(NonlinearMethod::picard, u, A);

    EXPECT_EQ(system.matrix_calls, 1);
    EXPECT_EQ(system.last_method, NonlinearMethod::picard);
    EXPECT_DOUBLE_EQ(system.last_time, 2.5);
    EXPECT_DOUBLE_EQ(system.last_derivative_shift, 2.0);
    expect_vector_equal(system.last_solution, u);
    expect_vector_equal(system.last_solution_derivative, Vector{2.0, 6.0});
}

TEST(BDF1Test, AssemblesNewtonMatrixWithBdf1DerivativeShift) {
    const BDF1 scheme;
    const TimeHistory history = make_history();
    RecordingSemiDiscreteSystem system;
    auto nonlinear_system = scheme.make_nonlinear_system(system, history, 2.25);
    const Vector u{2.0, 7.0};
    SparseMatrix A(2, 2, SparseStorageOrder::csc);

    nonlinear_system->assemble_matrix(NonlinearMethod::newton, u, A);

    EXPECT_EQ(system.matrix_calls, 1);
    EXPECT_EQ(system.last_method, NonlinearMethod::newton);
    EXPECT_DOUBLE_EQ(system.last_time, 2.25);
    EXPECT_DOUBLE_EQ(system.last_derivative_shift, 4.0);
    expect_vector_equal(system.last_solution_derivative, Vector{4.0, 12.0});
}

TEST(BDF1Test, RejectsInvalidTimeSteps) {
    const BDF1 scheme;
    const TimeHistory history = make_history();
    const RecordingSemiDiscreteSystem system;

    EXPECT_THROW(scheme.make_nonlinear_system(system, history, 2.0), std::invalid_argument);
    EXPECT_THROW(scheme.make_nonlinear_system(system, history, 1.0), std::invalid_argument);
    EXPECT_THROW(
        scheme.make_nonlinear_system(system, history, std::numeric_limits<double>::quiet_NaN()),
        std::invalid_argument);
    EXPECT_THROW(
        scheme.make_nonlinear_system(system, history, std::numeric_limits<double>::infinity()),
        std::invalid_argument);
}

TEST(BDF1Test, RejectsEmptyHistory) {
    const BDF1 scheme;
    const TimeHistory history;
    const RecordingSemiDiscreteSystem system;

    EXPECT_THROW(scheme.make_nonlinear_system(system, history, 1.0), std::logic_error);
}

TEST(BDF1Test, RejectsNonfinitePreviousTime) {
    const BDF1 scheme;
    const RecordingSemiDiscreteSystem system;
    TimeHistory nonfinite_history;
    nonfinite_history.accept({.t = std::numeric_limits<double>::infinity(), .u = Vector{1.0}});

    EXPECT_THROW(scheme.make_nonlinear_system(system, nonfinite_history, 1.0), std::logic_error);
}

TEST(BDF1Test, RejectsSolutionWithDifferentSizeFromHistory) {
    const BDF1 scheme;
    const TimeHistory history = make_history();
    RecordingSemiDiscreteSystem system;
    auto nonlinear_system = scheme.make_nonlinear_system(system, history, 2.5);
    const Vector u{2.0};
    Vector r;

    EXPECT_THROW(nonlinear_system->assemble_residual(u, r), std::invalid_argument);
}

} // namespace
