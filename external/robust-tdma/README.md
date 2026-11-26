# tdma

A header-only C++20 library for real tridiagonal systems. No external dependencies;
supported scalar types are `float` and `double`.

- `StandardSolver<T>`: classical sweep without pivoting, for matrices known to
  permit it, for example strictly row-diagonally dominant matrices.
- `PivotingSolver<T>`: Samarskii--Nikolaev nonmonotone sweep, the general choice
  for nonsingular tridiagonal matrices.

Both prepare the matrix once in O(n), then solve each right-hand side in O(n).

## Usage

Pass the actual matrix entries: the main diagonal has `n >= 1` elements and each
off-diagonal has `n - 1`. Arrays and vectors convert to the `std::span` arguments.

```cpp
#include <array>
#include <tdma/tdma.hpp>

int main() {
    const std::array lower{1.0};
    const std::array diagonal{0.0, 0.0};
    const std::array upper{1.0};
    const std::array rhs{3.0, 2.0};

    tdma::PivotingSolver<double> solver;
    solver.compute({lower, diagonal, upper});
    auto x = solver.solve(rhs); // (2, 3); allocates the returned vector.

    const std::array next_rhs{5.0, 4.0};
    solver.solve(next_rhs, x); // (4, 5); reuses the output, no allocation.
}
```

The public operations are `compute(matrix)`, `solve(rhs)`, `solve(rhs, output)`,
and `size()`. Output must already have exactly `size()` elements.
An empty solver has size zero. Both solvers own their prepared factors:
after `compute`, the source diagonals may be changed or destroyed.

Exact RHS/output aliasing is supported, so `solver.solve(b, b)` replaces a mutable
RHS with its solution. Partial overlap is not supported. `StandardSolver` uses
the output for intermediate values; its `solve` methods are `const` and allow
concurrent calls with separate outputs. `PivotingSolver` changes internal scratch;
use separate objects for simultaneous calls from different threads.
Ordinary copying and moving are supported.

Matrix dimensions and finite coefficients are checked during `compute`.
Invalid dimensions or NaN/infinity inputs throw `std::invalid_argument`;
zero/nonfinite pivots or factors throw `std::runtime_error` with a zero-based step.
Preparation writes directly into the solver. Preparing another matrix of the same
size reuses existing buffers without allocation; changing the size allocates fresh
buffers. Failed preparation invalidates the factorization (`size()` becomes zero);
call `compute` successfully again before solving. Solving before successful
preparation throws `std::logic_error`; wrong vector sizes throw `std::invalid_argument`.

`solve` checks only readiness and sizes before entering its arithmetic loops.
It does not scan RHS/results for finite values, estimate conditioning, or compute
residuals. Supply finite RHS values and assess accuracy as needed by your problem.
Nonsingularity alone does not guarantee accurate floating-point results for an
ill-conditioned or badly scaled system. See [the algorithms](docs/algorithms.md).

## Connecting to another project

Copy `include/tdma/` to your project's include path, or use CMake:

```cmake
add_subdirectory(external/robust-tdma)
target_link_libraries(my_program PRIVATE tdma::tdma)
```

The CMake target supplies the include path and C++20 requirement. There is no binary
library to build or link, and no compiler optimization flags are imposed on your
project. CMake 3.20 or newer is sufficient; examples are disabled by default.

## Examples

- [quickstart.cpp](examples/quickstart.cpp): classical sweep, then another RHS
  using the same matrix and output buffer.
- [pivoting.cpp](examples/pivoting.cpp): a nonsingular matrix with a zero first
  pivot, solved by the nonmonotone sweep.
- [heat_1d.cpp](examples/heat_1d.cpp): backward Euler for the 1D heat equation,
  preparing one matrix for all time steps and comparing with an analytic solution.

Build and run all three:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTDMA_BUILD_EXAMPLES=ON
cmake --build build --parallel
./build/tdma_quickstart
./build/tdma_pivoting
./build/tdma_heat_1d
```

Or compile an example directly:

```sh
c++ -std=c++20 -O3 -Iinclude examples/quickstart.cpp -o /tmp/tdma-quickstart
/tmp/tdma-quickstart
```

## License

[MIT](LICENSE).
