#include <Eigen/Sparse>
#include <Eigen/UmfPackSupport>
#include <gtest/gtest.h>
#include <vector>

TEST(UmfPackTest, SolvesSmallSparseSystem) {
    // A = [  4  -1   0 ]
    //     [ -1   4  -1 ]
    //     [  0  -1   3 ]
    // b = [15, 10, 10]^T
    // x = [5, 5, 5]^T

    constexpr int kMatrixSize = 3;
    Eigen::SparseMatrix<double> a(kMatrixSize, kMatrixSize);

    const std::vector<Eigen::Triplet<double>> triplets = {{0, 0, 4.0}, {0, 1, -1.0}, {1, 0, -1.0},
                                                          {1, 1, 4.0}, {1, 2, -1.0}, {2, 1, -1.0},
                                                          {2, 2, 3.0}};
    a.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::VectorXd b(kMatrixSize);
    b << 15.0, 10.0, 10.0;

    Eigen::UmfPackLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(a);

    ASSERT_EQ(solver.info(), Eigen::Success) << "UMFPACK factorization failed";

    const Eigen::VectorXd x = solver.solve(b);

    ASSERT_EQ(solver.info(), Eigen::Success) << "UMFPACK solve failed";

    const Eigen::VectorXd x_expected = Eigen::VectorXd::Constant(kMatrixSize, 5.0);
    constexpr double kTolerance = 1e-9;

    EXPECT_TRUE(x.isApprox(x_expected, kTolerance))
        << "Expected solution: [5, 5, 5], but got: [" << x.transpose() << "]";
}
