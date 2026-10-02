# LIN-001: проверка Eigen и UMFPACK

UMFPACK подключён через порт `suitesparse-umfpack` системы vcpkg и вызывается оболочкой
`Eigen::UmfPackLU`.

Проверка решает разреженную систему

$$
\begin{bmatrix}
4 & -1 & 0\\
-1 & 4 & -1\\
0 & -1 & 3
\end{bmatrix}
\begin{bmatrix}
x_1\\x_2\\x_3
\end{bmatrix}
=
\begin{bmatrix}
15\\10\\10
\end{bmatrix},
$$

точное решение которой равно $(5,5,5)$. Матрица хранится в разреженном формате.

Эта проверка подтверждает доступность Eigen, SuiteSparse и UMFPACK в сборке. Рабочий код
использует общий интерфейс `LinearSolver` и реализацию `UmfpackLinearSolver`.
