#include <gtest/gtest.h>
#include <petscsys.h>

int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    PetscCall(PetscInitialize(&argc, &argv, nullptr, nullptr));
    const int result = RUN_ALL_TESTS();
    PetscCall(PetscFinalize());
    return result;
}
