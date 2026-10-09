
\subsection{Two-Dimensional Linear Multiscale Elliptic Problem with a Non-Product Coefficient}
\label{sec:2d-nonproduct}

In this subsection we consider a two-dimensional problem ($\(d=2\)$) on the unit square
$\Omega=(0,1)^2$ with periodic boundary conditions on opposite faces.
The additive constant is fixed by $\int_\Omega u_\varepsilon\,dx=0$.
The governing equation takes the same form as~\eqref{ex_2D}, but the diffusion
coefficient is no longer factorizable as a single product $\(b(x)c(y)\)$.
Specifically,
\begin{equation}
\begin{aligned}
% AUTHOR QUERY: as written, the coefficient below can attain zero on Omega x Y;
% confirm whether the leading constant should be 4, or provide the intended
% uniformly elliptic coefficient and regenerated numerical results.
A\((x,y)\) &= 5 + \sin(2\pi x_1) + \cos(2\pi x_2) + \sin(2\pi y_1)\sin(2\pi y_2),\\
F(x) &= 30\sin(2\pi x_1) + 20\cos(2\pi x_2),
\end{aligned}
\end{equation}
where $y=(y_1,y_2)=(x_1/\varepsilon,x_2/\varepsilon)$ is the fast variable.
Both $A$ and $F$ are finite sums of tensor-product terms, so they remain
compatible with the tensorized quadrature assumption
\eqref{eq:coefficient-tensor-format}. The relevant distinction from the
previous benchmark is not a lack of tensor structure: because $A$ cannot be
written as one product $\(b(x)c(y)\)$, the slow-variable factor does not cancel
from the cell equations, and the cell functions generally depend on $x$. The
factor $\sin(2\pi y_1)\sin(2\pi y_2)$ is itself a rank-one tensor product in
the two fast coordinates.

The zero-mean condition $\int_\Omega F\,\mathrm{d}x=0$ holds, ensuring
compatibility of the periodic equation. The finite element reference solution
is computed using the OpenPFEM-based solver with the same periodic boundary
conditions, and the zero-mean comparison is
defined by \eqref{eq:numerical_physical_mean}.
As written, this coefficient can attain zero on $\Omega\times Y$, so it
does not satisfy the uniform ellipticity assumption of the convergence
theorems. This example is therefore retained as a numerical test, not as
a verification of those theorems; the coefficient specification needs to
be confirmed before such a claim can be made.

As in the product-form case, the corrector hierarchy for the present problem follows
the structure
\[
U_1 = u_0 + \varepsilon(\hat u_1 + \tilde u_1),\qquad
U_2 = U_1 + \varepsilon^2(\hat u_2 + \delta u_2 + \tilde u_2).
\]
The key difference from the previous example is that the cell functions now
depend on both the slow and fast variables, even though the coefficient and source term have low
tensor rank. Table~\ref{tab:2dns-network-selection} reports the final network
architectures and the corresponding final loss values for each quantity in the
expansion hierarchy. For the oscillatory correctors $\hat u_2$ and $\hat u_3$,
the loss values are noticeably higher than those in the product-form case,
which reflects the increased difficulty of resolving the slow-dependent cell
hierarchy.
The $L^2$ errors are reported on six scales,
\[
\varepsilon \in \{1/3,1/5,1/8,1/10,1/12,1/15\},
\]
whereas the $H^1$ semi-norm errors are reported on the five scales
$\{1/3,1/5,1/8,1/10,1/15\}$.

Figure~\ref{fig:2dns-eps10-2x2} displays the spatial distribution of the
approximation errors for the two-dimensional non-product problem at
$\varepsilon=1/10$. Panel (a) shows the second-order TNN approximation $U_2$,
while panels (b)--(d) show the error fields at different truncation orders.
The error maps show the additional features produced by the slow-dependent
cell functions in the corrector components.

Tables~\ref{tab:2dns-\(L^2\)-errors} and~\ref{tab:2dns-\(H^1\)-errors} collect the
$L^2$ errors and $H^1$ semi-norm errors, respectively.

In the $L^2$ norm, a least-squares fit on the six reported scales yields
convergence rates of approximately $\(0.977\)$ for $u_0$, $\(2.02\)$ for $U_1$, $\(2.72\)$
for $U_2$, and $\(2.56\)$ for $U_3^*$. The first two rates agree closely with the
nominal orders $1$ and $2$; the two higher-order curves show rates below the
nominal third order on the reported scales. The third-order partial correction
still reduces the error at every listed scale.

In the $H^1$ semi-norm, the first-order partial approximation $U_1^*$ achieves a convergence rate of approximately $\(0.987\)$, consistent with the expected first-order rate. For the second-order partial approximation $U_2^*$, the least-squares fit over all five scales gives a rate of approximately $\(1.13\)$, which is below the ideal second-order rate; this discrepancy is associated with the finest scale $\varepsilon=1/15$, where the $H^1$ error of $U_2^*$ exhibits a slight increase relative to the value at $\varepsilon=1/10$ (cf. Table~\ref{tab:2dns-\(H^1\)-errors}). Fitting the first four scales only yields a rate of approximately $\(1.51\)$.

Figure~\ref{fig:2d-ex3-errors} presents the convergence behavior of the TNN approximations in the $L^2$ norm and $H^1$ semi-norm for the non-product problem. The $L^2$ data show approximately first-, second-, and third-order convergence for $u_0$, $U_1$, and $U_2$, respectively, while the third-order partial approximation $U_3^*$ further reduces the error. In the $H^1$ semi-norm, $U_1^*$ converges at approximately first order, and $U_2^*$ reduces the error relative to $U_1^*$ on all reported scales, although a deviation from the ideal second-order rate is observed on the finest scale.

In the implementation of this non-product example, a significant computational challenge arises from the construction of the stiffness matrix for $\tilde u_2$. Unlike the cell correctors that live on the fast-variable cell $Y$, the second-order corrector $\tilde u_2$ is a function of the slow variables $x=(x_1,x_2)$ only; the fast-variable dependence has already been eliminated by integrating over $Y$ when forming its governing equation. Consequently, the stiffness matrix entries for $\tilde u_2$ contain coefficients such as $A_0^{ij}(x)$ (cf. Eq.~(13)), each of which is itself defined as an integral over $Y$ and must be evaluated numerically. Although the TNN ansatz enables a factorized evaluation of one-dimensional integrals, the non-product coefficient $A(x,y)$ contains multiple tensor-product terms, and the nonlinear cell corrector $\chi$ further increases the effective rank of the trial functions. The coefficient blocks $A_0^{11},A_0^{12},A_0^{21},A_0^{22}$ therefore have TNN ranks $p_A+p_Ap_\chi$ and $p_Ap_\chi$, respectively, and the trial rank $p_{\tilde u_2}$ is multiplied into each one when the eight bilinear forms of Eq.~(6-58) are assembled. The resulting factorized tensor is very large (its total rank is on the order of several hundred), and the number of one-dimensional sub-integrals grows proportionally. We therefore adopt a direct two-dimensional quadrature on the Cartesian-product grid with $800\times 800 = 6.4\times 10^{5}$ nodes in $Y$ to evaluate these coefficient integrals.

Even with the direct quadrature strategy, the naive assembly of the stiffness-matrix entry
\[
T_{ab} = \sum_{p,j}\alpha_p\alpha_j\prod_{d=1}^{2}\Bigl(\sum_{n=1}^{N}w^{(1)}_n\phi^{(d)}_{p,a,n}\phi^{(d)}_{j,b,n}\Bigr)
\]
requires forming the intermediate tensor $S[d,p,j,a,b]=\sum_n w^{(1)}_n\phi^{(d)}_{p,a,n}\phi^{(d)}_{j,b,n}$ of shape $\mathbb{R}^{2\times p_\text{eff}\times p_\text{eff}\times N\times N}$, whose size scales as $p_\text{eff}^2 N^2$ and would dominate GPU memory in the present configuration. This is avoided by a memory-efficient reformulation
\[
T = FF^\top,\quad F_{a,n_1,n_2}=\sum_i\alpha_i\,\sqrt{w^{(1)}_{n_1}}\,\phi^{(0)}_{i,a,n_1}\,\sqrt{w^{(1)}_{n_2}}\,\phi^{(1)}_{i,a,n_2},
\]
implemented as a custom \texttt{torch.autograd.Function} (\texttt{MyMatrixFunction}) with a closed-form backward pass. This reduces the memory footprint by a factor of order $p_\text{eff}^2$ and achieves roughly a $30\times$ speedup over the naive assembly. An additional structural advantage of the TNN over a standard feedforward network with two-dimensional input is that the two-dimensional function values at the $Y$-quadrature nodes are generated from one-dimensional sub-network outputs, requiring far fewer forward evaluations.

\begin{figure}[htbp]
  \centering
  % 左子图 (a)
  \begin{minipage}{0.48\textwidth}
    \centering
    \includegraphics[width=\linewidth]{./2d_ex3/l2_error.png}
  \end{minipage}
  \hfill % 两个子图之间留出空隙
  % 右子图 (b)
  \begin{minipage}{0.48\textwidth}
    \centering
    \includegraphics[width=\linewidth]{./2d_ex3/h1_error.png}
  \end{minipage}
  % 整张图的总标题
  \caption{Two-dimensional non-product periodic problem: log--log scale error plots against $\varepsilon$ for successive TNN approximations. Left: $L^2$ error; Right: $H^1$ semi-norm error.}
  \label{fig:2d-ex3-errors}
\end{figure}
