#include "petsc_test_session.hpp"

#include <gtest/gtest.h>
#include <petscsystypes.h>
#include <petsctao.h>
#include <petscvec.h>

/**
 * Callback-функция вычисления невязок для задачи наименьших квадратов:
 * F_0(a, b) = a - 1
 * F_1(a, b) = b - 2
 * F_2(a, b) = a + b - 3
 *
 * Точное решение: a = 1.0, b = 2.0 (вектор невязок F = [0, 0, 0]^T).
 */
static PetscErrorCode evaluate_residual(Tao tao, Vec x, Vec f, void* ctx) {
    (void)tao;
    (void)ctx;
    const PetscScalar* x_arr = nullptr;
    PetscScalar* f_arr = nullptr;

    PetscCall(VecGetArrayRead(x, &x_arr));
    PetscCall(VecGetArray(f, &f_arr));

    const PetscReal a = x_arr[0];
    const PetscReal b = x_arr[1];
    f_arr[0] = a - 1.;
    f_arr[1] = b - 2.;
    f_arr[2] = a + b - 3.;

    PetscCall(VecRestoreArrayRead(x, &x_arr));
    PetscCall(VecRestoreArray(f, &f_arr));

    return PETSC_SUCCESS;
}

TEST(PetscTaoOptimization, PoundersLeastSquaresToyProblem) {
    ensure_petsc_test_session();

    // Создание вектора параметров x = (a, b) размера 2
    Vec x = nullptr;
    ASSERT_EQ(VecCreateSeq(PETSC_COMM_SELF, 2, &x), PETSC_SUCCESS);

    // x0 = (0.0, 0.0)
    ASSERT_EQ(VecSetValue(x, 0, 0.0, INSERT_VALUES), PETSC_SUCCESS);
    ASSERT_EQ(VecSetValue(x, 1, 0.0, INSERT_VALUES), PETSC_SUCCESS);
    ASSERT_EQ(VecAssemblyBegin(x), PETSC_SUCCESS);
    ASSERT_EQ(VecAssemblyEnd(x), PETSC_SUCCESS);

    // Создание вектора невязок размер 3
    Vec f = nullptr;
    ASSERT_EQ(VecCreateSeq(PETSC_COMM_SELF, 3, &f), PETSC_SUCCESS);

    // Создание и настройка TAO
    Tao tao = nullptr;
    ASSERT_EQ(TaoCreate(PETSC_COMM_SELF, &tao), PETSC_SUCCESS);
    ASSERT_EQ(TaoSetType(tao, TAOPOUNDERS), PETSC_SUCCESS);
    ASSERT_EQ(TaoSetSolution(tao, x), PETSC_SUCCESS);
    ASSERT_EQ(TaoSetResidualRoutine(tao, f, evaluate_residual, nullptr), PETSC_SUCCESS);
    ASSERT_EQ(TaoSetFromOptions(tao), PETSC_SUCCESS);

    // Запуск оптимизации
    ASSERT_EQ(TaoSolve(tao), PETSC_SUCCESS);

    // Проверка сходимости
    TaoConvergedReason reason = TAO_CONTINUE_ITERATING;
    ASSERT_EQ(TaoGetConvergedReason(tao, &reason), PETSC_SUCCESS);
    EXPECT_GT(reason, 0) << "TAO solver did not converge. Reason code: " << reason;

    // Проверка найденного решения
    const PetscScalar* sol_arr = nullptr;
    ASSERT_EQ(VecGetArrayRead(x, &sol_arr), PETSC_SUCCESS);
    const PetscReal a_opt = sol_arr[0];
    const PetscReal b_opt = sol_arr[1];
    ASSERT_EQ(VecRestoreArrayRead(x, &sol_arr), PETSC_SUCCESS);

    EXPECT_NEAR(a_opt, 1.0, 1e-4);
    EXPECT_NEAR(b_opt, 2.0, 1e-4);

    // Освобождение ресурсов
    ASSERT_EQ(TaoDestroy(&tao), PETSC_SUCCESS);
    ASSERT_EQ(VecDestroy(&x), PETSC_SUCCESS);
    ASSERT_EQ(VecDestroy(&f), PETSC_SUCCESS);
}
