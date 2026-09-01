#pragma once

#include <petscsys.h>
#include <stdexcept>

class PetscTestSession {
public:
    PetscTestSession() {
        PetscBool initialized = PETSC_FALSE;
        if (PetscInitialized(&initialized) != PETSC_SUCCESS) {
            throw std::runtime_error("Failed to query PETSc initialization state");
        }
        if (initialized == PETSC_FALSE) {
            if (PetscInitializeNoArguments() != PETSC_SUCCESS) {
                throw std::runtime_error("Failed to initialize PETSc for tests");
            }
            owns_petsc_ = true;
        }
    }

    ~PetscTestSession() {
        PetscBool finalized = PETSC_FALSE;
        if (owns_petsc_ && PetscFinalized(&finalized) == PETSC_SUCCESS &&
            finalized == PETSC_FALSE) {
            PetscFinalize();
        }
    }

    PetscTestSession(const PetscTestSession&) = delete;
    PetscTestSession& operator=(const PetscTestSession&) = delete;

private:
    bool owns_petsc_ = false;
};

inline void ensure_petsc_test_session() {
    static PetscTestSession session;
}
