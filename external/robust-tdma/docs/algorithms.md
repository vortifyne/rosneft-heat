# Algorithms

For `n >= 1`, the matrix is stored as `lower[i] = A(i+1,i)`,
`diagonal[i] = A(i,i)`, `upper[i] = A(i,i+1)`.
Both solvers prepare matrix-dependent factors once, then reuse them for every RHS.
Preparation, solving, and storage have O(n) complexity.

## Classical sweep

Write the diagonals as l, d, u. The LU recurrence is

```text
p[0] = d[0]
m[i-1] = l[i-1] / p[i-1]
p[i] = d[i] - m[i-1]*u[i-1]          (i = 1, ..., n-1)
```

Store u, m and inverse pivots. For each RHS f:

```text
z[0] = f[0]
z[i] = f[i] - m[i-1]*z[i-1]         (i = 1, ..., n-1)

x[n-1] = z[n-1] / p[n-1]
x[i] = (z[i] - u[i]*x[i+1]) / p[i]  (i = n-2, ..., 0)
```

The output vector holds z during the forward sweep, then x during back substitution.
Exact RHS/output aliasing is allowed. Every pivot must be nonzero.
Strict row diagonal dominance is sufficient, as is irreducible weak row dominance
with at least one strict row; these are sufficient conditions, not necessary ones.

## Nonmonotone sweep

The Samarskii--Nikolaev notation uses

```text
-a[i]*y[i-1] + c[i]*y[i] - b[i]*y[i+1] = f[i]
c[i] = diagonal[i], a[i+1] = -lower[i], b[i] = -upper[i].
```

The first two reduced equations at step i are

```text
 C*y[m] - b[i]*y[i+1] = F
-A*y[m] + c[i+1]*y[i+1] - b[i+1]*y[i+2] = Phi.
```

Initially C = c[0], A = a[1], F = f[0], Phi = f[1], m = 0.
Choose the larger of |C| and |b[i]| as denominator.
The relation stored at each step is `y[theta[i]] = alpha[i]*y[chi[i]] + beta[i]`:

| Quantity | Regular: abs(C) >= abs(b[i]) | Pivot: abs(C) < abs(b[i]) |
|---|---|---|
| alpha[i] | b[i]/C | C/b[i] |
| beta[i] | F/C | -F/b[i] |
| next C | c[i+1] - A*alpha[i] | c[i+1]*alpha[i] - A |
| next F | Phi + A*beta[i] | Phi - c[i+1]*beta[i] |
| theta[i] | m | i+1 |
| chi[i] (next m) | i+1 | m |
| next A | a[i+2] | a[i+2]*alpha[i] |
| next Phi | f[i+2] | f[i+2] + a[i+2]*beta[i] |

At the last step (i = n-2), next A and next Phi are unused: there is no access to
`lower[n-1]` or `rhs[n]`. For n = 1, solve the scalar equation directly.

The prepared representation stores alpha, chi and coefficients s, q, r:

```text
regular: (s, q, r) = ( 1/C,  A,       0)
pivot:   (s, q, r) = (-1/b, -c[i+1], a[i+2])
```

For a new RHS, compute `beta[i] = s[i]*F`,
`F = Phi + q[i]*beta[i]`, and, except at the last step,
`Phi = f[i+2] + r[i]*beta[i]`. Pivot comparisons are not repeated.
After the final step set `y[chi[n-2]] = F/C` and reconstruct the remaining unknowns
in reverse step order.

Only chi is stored: a regular step has `chi[i] == i+1`, so its theta is
`i == 0 ? 0 : chi[i-1]`; otherwise theta is i+1. All RHS entries are consumed before
reconstruction, which also permits exact RHS/output aliasing with separate beta scratch.

For a nonsingular matrix in exact arithmetic, C and b cannot both vanish at a step,
and the final C is nonzero. The chosen denominator gives |alpha| <= 1, bounding
amplification in reconstruction. This establishes algebraic applicability; it is
not a bound on the condition number or all floating-point errors.
Zero off-diagonal entries and block-triangular systems need no special splitting.

## Implementation and accuracy

Preparation validates dimensions and coefficients as they are read or computed,
writing directly into owned vectors. Another matrix of the same size reuses these
buffers; a different size allocates fresh buffers.
A failed preparation leaves the solver unprepared (`size() == 0`); its buffers remain
available for the next `compute`, which must succeed before any solve.
Both methods store inverse pivots. Zero pivots, nonfinite factors and reciprocal
overflow are reported, with no arbitrary small-pivot threshold.

Each solve uses the prepared factors without allocation when the caller supplies
output. The classical sweep uses output for intermediate values; the nonmonotone
sweep needs a separate buffer to retain intermediate values until reconstruction.
Readiness and size checks occur before the arithmetic loops.
No RHS scans, residual calculations, or per-operation validation occur there.

The heat example uses central differences and backward Euler for u_t = kappa*u_xx.
With zero boundary values, the interior system has diagonals
`(-r, 1+2r, -r)`, r = kappa*dt/h², and the previous time layer as RHS.
Its error against `exp(-kappa*pi²*t)*sin(pi*x)` includes O(dt + h²) discretization
error and is not a direct measure of the tridiagonal solver's rounding error.

## Source

A. A. Samarskii and E. S. Nikolaev, *Methods for Solving Grid Equations*,
Nauka, Moscow, 1978: Chapter II, §1, pp. 73–80, and §2.4, pp. 93–97.
The [scanned book](https://samarskii.ru/books/book1978.pdf) is a reference;
using the library does not require downloading it.
