#pragma once

// manual: the caller marks reuse boundaries with reset_linear_solver().
enum class PreconditionerReuse {
    none,
    step,
    manual
};
