# DDM с матрицей Пикара

## План реализации

1. `numerics/optimization` — шаг Гаусса–Ньютона с проекцией на границы, градиент и оценка гессиана. `basin/inverse` — один прямой расчёт, интерполяция в скважины, взвешивание невязки и якобиана.
2. `BasinHeatParameterSpace` — базальный поток и его производные по нормированным параметрам в заданный момент.
3. `basin/forward` — стационарный старт и чувствительности после каждого принятого шага. `basin/discretization` — правые части DDM. `numerics` — решение с последней матрицей Пикара для всех правых частей, при сходимости за ноль итераций — одна сборка.
4. `basin/discretization` — чувствительности температуры на гранях. `basin/physics/easy_ro` — накопление производных интегралов реакций. `basin/mesh` — перенос чувствительностей при смене сетки.
5. `tests/basin` — сверка направленной производной скважинных невязок с конечной разностью при постоянных теплофизических коэффициентах.

## Гаусс–Ньютон · один прямой расчёт в точке $\eta$

Строки плотного якобиана соответствуют наблюдениям температуры и $R_o$ в порядке входных данных, столбцы — нормированным параметрам $\eta$.

$$
r=d^{\rm calc}-d^{\rm obs},\qquad J\simeq\partial_{\eta} d^{\rm calc},\qquad
W=\operatorname{diag}(1/\sigma_k),\qquad r_w=Wr,\quad J_w=WJ.
$$

$$
\Phi=\tfrac12 r_w^\top r_w,\qquad g=J_w^\top r_w,\qquad H=J_w^\top J_w,
\qquad H p=-g,\quad \eta^+=\operatorname{clip}(\eta+p,0,1).
$$

В отчёте обоих оптимизаторов сохраняется $\Phi$.

## Прямой расчёт · один принятый шаг BDF1

Сначала температура до сходимости, затем по одной правой части на параметр. Звёздочка — последняя сборка Пикара.

$$
S_j\simeq\partial_{\eta_j}T,\qquad b_j(t)=\partial_{\eta_j}q(t),\qquad
e=-\left.\partial_q R_{\rm heat}\right|_{T\ \text{фикс.}}
\quad\text{(базальный приток и граничные слагаемые в реконструкции потоков)} .
$$

$$
\begin{aligned}
K_0^* S_{0,j} &= e_0 b_j(t_0) && \text{(стационарный старт)},\\
A_n^* S_{n,j} &= \frac{M_n^*}{\Delta t_n}S_{n-1,j}
                    + e_n b_j\!\left(\tfrac{t_{n-1}+t_n}{2}\right),\\
A_n^* &= K_n^*+M_n^*/\Delta t_n,\qquad
M_n^*=\operatorname{diag}\!\left(|P_n|\,(\rho c)_{P,n}^*\right).
\end{aligned}
$$

## EASY%Ro · история материальной точки

$$
\beta_\ell=E_\ell/R_g,\quad k_\ell(T)=A e^{-\beta_\ell/T},\quad
a=\min(t_n,\max(t_{n-1},t_{\rm dep})),\quad s_j\simeq\partial_{\eta_j}T_\sigma .
$$

Температура грани и её чувствительность линейны по времени внутри шага.

$$
I_{\ell,n}=I_{\ell,n-1}+\int_a^{t_n}k_\ell(T_\sigma)\,dt,\qquad
Z_{\ell j,n}=Z_{\ell j,n-1}+
\int_a^{t_n}k_\ell(T_\sigma)\frac{\beta_\ell}{T_\sigma^2}\,s_j\,dt,
\qquad I_{\ell,0}=Z_{\ell j,0}=0 .
$$

$$
R_o=\exp\!\left[-1.6+3.7\sum_\ell f_\ell(1-e^{-I_\ell})\right],\qquad
U_j=3.7R_o\sum_\ell f_\ell e^{-I_\ell}Z_{\ell j}\simeq\partial_{\eta_j}R_o .
$$

## Смена сетки и финальные наблюдения

$$
S_j^+=P_T S_j^-,\qquad Z_{\ell j}^+=P_I Z_{\ell j}^-,
\qquad S_j^+=0\ \text{при задании }T_{\rm surf},\quad Z_{\ell j}^+=0\ \text{при сбросе }I_\ell .
$$

$$
r_k=d_k^{\rm calc}-d_k^{\rm obs},\qquad
J_{kj}=
\begin{cases}
B_k^T[S_{j};s_j]_{t_f}, & T,\\
B_k^R U_j(t_f), & R_o .
\end{cases}
\qquad (B_k^T,B_k^R\text{ — веса интерполяции в скважину})
$$

## Параметризация потока и реконструкция на грани

$$
q(t)=\sum_{i=0}^{m-1}\psi_i(t)q_i,\quad b_j(t)=\sum_i\psi_i(t)Q_{ij}
\qquad(\psi_i\text{ — кусочно-линейный узловой базис}).
$$

$$
q_i=q_{\min}+(q_{\max}-q_{\min})\prod_{k=0}^{i}\eta_k,\qquad
Q_{ij}=\begin{cases}
(q_{\max}-q_{\min})\displaystyle\prod_{0\le k\le i,\ k\ne j}\eta_k, & j\le i,\\
0, & j>i.
\end{cases}
$$

$$
r_P=d_{P\sigma}/\lambda_P,\quad r'_{Pj}=-r_P(\partial_T\lambda_P/\lambda_P)S_{P,j},
\quad w=\frac{r_P}{r_P+r_N},\quad
w'_j=\frac{r_N r'_{Pj}-r_P r'_{Nj}}{(r_P+r_N)^2}.
$$

$$
s_{\sigma,j}=(1-w)S_{P,j}+w S_{N,j}+(T_N-T_P)w'_j,\quad
s_{{\rm surf},j}=0,\quad s_{{\rm bot},j}=S_{P,j}+r_P b_j(t)+q(t)r'_{Pj}.
$$
