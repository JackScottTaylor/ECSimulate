# ECSimulate Simulation Theory

This document explains all of the theory behind how ECSimulate performs its various simulations. Every aspect should be covered with a rigorous yet understandable explanation.

# Discretisation
ECSimulate is a numerical simulation package and does not consider analytical solutions, therefore all functions are considered discretely. 

Concentration is modelled as a time-dependant vector over a set number of points. We will therefore adopt the notation $C_i(t)$ to denote the the concentration at the $i^\text{th}$ position at time $t$.

To exactly update the concentration at each spatial point, the following would have to be solved analytically.

$$
\begin{split}
C_i(t+\Delta t) &= C_i(t) + \int_0^{\Delta t}\frac{d C_i(t + T)}{dT} dT \\
&= C_i(t) + \Delta C_i(t, \Delta t)
\end{split}
$$

However this is not possible exactly in a fully discrete simulation therefore the aim is to be able to accurately estimate the update function, $\Delta C_i(t, \Delta t)$.

## Explicit Approximation
The perhaps most intuitive way to approximate the update function is to calculate the time derivative at the initial timepoint, and assume it does not change over the timestep , $\Delta t$. This can be summarised:

$$
\Delta C_i(t, \Delta t) \approx \Delta t \frac{dC_i(t)}{dt}
$$

This is the fully explicit method. It is known however that updating in this manner is not numerically stable, and can lead to unphysical values extremely quickly.

If modelling concentration for example, then a suitably large time-derivative may lead to negative values, which is of course non-physical.

This approximation is only suitable if small-enough time-steps can be used such that the change per step is relatively small and the time-derivative does not change significantly over that time.

## Implicit Approximation
A very similar but slightly less intuitive approximation is to not use the derivative calculated at the current time-step, but instead the future one.

$$
\Delta C_i(t, \Delta t) \approx \Delta t \frac{dC_i(t + \Delta t)}{dt}
$$

This method is often called the Backward-Euler method. Unlike the fully explicit approximation, this method is unconditionally stable, i.e. the numerical solutions will never explode to infinity.

It is worth mentioning however that unconditional stability does not also imply accuracy. This method is known to often damp solutions such that changes are not as large as expected. However, as with the explicit formulation, decreasing the time-step should converge to the exact solution.

Depending on how the time-derivative is calculated, calculating the future derivative may be straightforward or require an iterative process. In which case it is reasonable to consider whether reducing the timesteps for the explicit aproximation would be less computationally expensive for the same level of accuracy.

## Variable Implicitness
We have already discussed that the explicit method often leads to exploding solutions, whereas the implicit method can lead to over-damped solutions. It is reasonable then to consider if there is a good middle ground. Here we consider still approximating the time-derivative over the time-step, but approximating it as a weighted average of the current, and future time-steps.

$$
\Delta C_i(t, \Delta t) \approx \Delta t \left( 
\theta\frac{dC_i(t + \Delta t)}{dt} + 
(1-\theta)\frac{dC_i(t)}{dt}
\right)
$$

By setting $\theta = 0.5$ the Crank-Nicholson formulation is found. Crank-Nicholson is also unconditionally stable and can often lead to more accurate results than the fully implicit method. 

Using Crank-Nicholson however can still lead to unwanted oscillations in the numerical solutions. The oscillations can be damped by increasing $\theta$. 

# Diffusion
The concentration, $C$, of a species changes over time according to the diffusion equation, with associated diffusion coefficient $D$.

$$
\frac{\partial C}{\partial t} = D \nabla^2 C
$$

However when dealing with a planar electrode, it is the 1D diffusion equation which is usually considered.

$$
\frac{\partial C}{\partial t} = D \frac{\partial^2 C}{\partial x^2}
$$

ECSimulate only considers 1D systems, so the following derivations concern only the 1D diffusion equation. 

## 1D Discrete Diffusion
Modelling the update to concentration at each time point according to any of the approximations to the update function described above relies on being able to approximate the second spatial derivative at each spatial position.

In the first instance we will consider an evenly spaced spatial grid such that the $i^\text{th}$ grid point corresponds to a distance $i\Delta x$ from the electrode boundary.

We can then use a centered first-oder approximation to the second-derivative:

$$
\frac{\partial^2 C_i}{\partial x^2} \approx \frac{C_{i-1} - 2C_i + C_{i+1}}{\Delta x ^2}
$$

We can now consider inserting this approximation into the variable implicitness approximation to obtain an estimate for the update function and obtain an equation describing how the concentration at each spatial and time point should change.

$$
\begin{split}
\Delta C_i(t, \Delta t) \approx D \Delta t \bigg(
\theta &\frac{C_{i-1}(t+\Delta t) - 2C_i(t+\Delta t) + C_{i+1}(t+\Delta t)}{\Delta x ^2} + \\
&+(1-\theta) \frac{C_{i-1}(t) - 2C_i(t) + C_{i+1}(t)}{\Delta x ^2}
\bigg)
\end{split}
$$

This update function is then used to describe how to update the concentration at each time step.

$$
\begin{split}
C_i(t + \Delta t) = C_i(t) +  D \Delta t \bigg(
\theta &\frac{C_{i-1}(t+\Delta t) - 2C_i(t+\Delta t) + C_{i+1}(t+\Delta t)}{\Delta x ^2} + \\
&+(1-\theta) \frac{C_{i-1}(t) - 2C_i(t) + C_{i+1}(t)}{\Delta x ^2}
\bigg)
\end{split}
$$

Although this formulation looks terrible to actually then go and solve, it is in fact not that bad. The next step is to move all the future time-step coefficients to the LHS and the current time-step to the RHS. We will also intriduce the constant $\phi = {D \Delta t}/{\Delta x^2}$.

$$
\begin{split}
(1 + 2\theta\phi)C_i(t+\Delta t) - &\theta\phi C_{i-1}(t+\Delta t) - 
\theta\phi C_{i-1}(t+\Delta t) = \\
= &(1 - 2(1-\theta)\phi)C_i(t) + (1-\theta)\phi C_{i-1}(t) + 
(1-\theta)\phi C_{i-1}(t)
\end{split}
$$

This can now be written in the form of a matrix equation, with one matrix acting on the vector of future time-step concentrations and another acting on the vector of current time-step concentrations.

$$
\textbf{A}\textbf{C}_{t+\Delta t} = \textbf{B}\textbf{C}_{t}
$$

$$
\begin{split}
&
\begin{pmatrix}
\ddots & \ddots & \ddots
\\
& -\theta \phi & 1 + 2\theta\phi & -\theta\phi
\\
&&\ddots & \ddots & \ddots
\end{pmatrix}
\begin{pmatrix}
\vdots \\
C_{i-1}(t+\Delta t) \\
C_{i}(t+\Delta t) \\
C_{i+1}(t+\Delta t) \\
\vdots
\end{pmatrix}
= \\& =
\begin{pmatrix}
\ddots & \ddots & \ddots
\\
& (1-\theta) \phi & 1 - 2(1-\theta)\phi & (1-\theta)\phi
\\
&&\ddots & \ddots & \ddots
\end{pmatrix}
\begin{pmatrix}
\vdots \\
C_{i-1}(t) \\
C_{i}(t) \\
C_{i+1}(t) \\
\vdots
\end{pmatrix}
\end{split}
$$

Now note that as written, the two matrices $\textbf{A}$ and $\textbf{B}$ are constants and do not need to be recomputed each step, hence obtaining the next step of concentrations becomes quite efficient.

The most obvious way to implemt this methodology is to calculate once the matrix $\textbf{A}^{-1}\textbf{B}$ as repeated left-multiplication by this matrix would iterate the concentrations. While this should in theory work, small floating-point errors may increase after large numbers of iterations leading to progressively less accurate simulations.

A useful point to note is that in this formulation, both $\textbf{A}$ and $\textbf{B}$ are tridiagonal matrices. An extremely efficient algorithm, "Thomas Algorithm", exists for solving these exact kind of equations and is the recommended method.

In ECSimulate, the `solve_banded` method from `scipy` is used, as this allows for matrix equations with a general number of bands to be solved, which is required as we go beyond just modelling diffusion.

## Boundary Conditions
Some readers may have realised at this point that the previous derivation does not consider suitable boundary conditions. In fact if we suppose that a total of $N$ spatial points are being simulated, with the first point being at $x=0$ and the last point being at $x = N\Delta x$ then the diffusion equation for $C_0$ requires $C_{-1}$ and likewise at $C_{N-1}$, $C_N$ would be required. Therefore considering the boundary conditions essentially translates in to how we model these values.

### Reflective Electrode Boundary
In `ECSimulate` we consider the left boundary ($x=0$) to be the planar electrode. When only considering diffusion this is treated as a reflective boundary, i.e. there is no flux across the boundary.

$$
\frac{\partial C}{\partial x}\bigg|_{x=0} = 0
$$

We can find a suitable value for $C_{-1}$ by approximating the gradient at the boundary using a centralised finite differences approach.

$$
\frac{\partial C}{\partial x}\bigg|_{x=0} \approx \frac{C_{1} - C_{-1}}{2 \Delta x}
$$

Substituting this approximation in to the exact boundary condition yields the following condition on $C_{-1}$.

$$
C_{-1} = C_1
$$

When we write out the previous diffusion derivation for $C_0$ and replace all instances of $C_{-1}$ with $C_1$ we are left with the following.

$$
\begin{split}
(1 + 2\theta\phi)C_0(t+\Delta t) - &2\theta\phi C_{1}(t+\Delta t) = 
(1 - 2(1-\theta)\phi)C_0(t) + 2(1-\theta)\phi C_{1}(t) 
\end{split}
$$

This is cleanly implemented in the top rows of the two banded diffusion matrices.

$$
\begin{split}
&
\begin{pmatrix}
1 + 2\theta \phi & -2\theta \phi
\\
 -\theta \phi & 1 + 2\theta\phi & -\theta\phi
\\
& -\theta \phi & 1 + 2\theta\phi & -\theta\phi
 \\
&\ddots & \ddots & \ddots
\end{pmatrix}
\begin{pmatrix}
C_{0}(t+\Delta t) \\
C_{1}(t+\Delta t) \\
C_{2}(t+\Delta t) \\
C_{3}(t+\Delta t) \\
\vdots
\end{pmatrix}
= \\
&=
\begin{pmatrix}
1 - 2(1-\theta) \phi & 2(1-\theta) \phi
\\
 (1-\theta) \phi & 1 - 2(1-\theta)\phi & (1-\theta)\phi
\\
& (1-\theta) \phi & 1 - 2(1-\theta)\phi & (1-\theta)\phi
 \\
&\ddots & \ddots & \ddots
\end{pmatrix}
\begin{pmatrix}
C_{0}(t) \\
C_{1}(t) \\
C_{2}(t) \\
C_{3}(t) \\
\vdots
\end{pmatrix}
\end{split}
$$

### Dirichlet Bulk Concentration Boundary
On the right hand side of the simulated system, we consider this position to be far enough away from the electrode that this is the bulk solution and that this concentration does not change over the course of the simulated experiment.

$$
C_{N-1}(t + \Delta t) = C_{N-1}(t)
$$

This condition is directly implementable in the last row of the diffusion matrices, which after combination with the reflective electrode boundary give the final form of the diffusion matrices used by `ECSimulate`.

$$
\begin{split}
&
\begin{pmatrix}
1 + 2\theta \phi & -2\theta \phi
\\
 -\theta \phi & 1 + 2\theta\phi & -\theta\phi
\\
& -\theta \phi & 1 + 2\theta\phi & -\theta\phi
 \\
&\ddots & \ddots & \ddots \\
&& -\theta \phi & 1 + 2\theta\phi & -\theta\phi \\
&&&&1
\end{pmatrix}
\begin{pmatrix}
C_{0}(t+\Delta t) \\
C_{1}(t+\Delta t) \\
C_{2}(t+\Delta t) \\
\vdots \\
C_{N-2}(t+\Delta t) \\
C_{N-1}(t+\Delta t)
\end{pmatrix}
= \\
&=
\begin{pmatrix}
1 - 2(1-\theta) \phi & 2(1-\theta) \phi
\\
 (1-\theta) \phi & 1 - 2(1-\theta)\phi & (1-\theta)\phi
\\
& (1-\theta) \phi & 1 - 2(1-\theta)\phi & (1-\theta)\phi
 \\
&\ddots & \ddots & \ddots
\\
&& (1-\theta) \phi & 1 - 2(1-\theta)\phi & (1-\theta)\phi \\
&&&&1
\end{pmatrix}
\begin{pmatrix}
C_{0}(t) \\
C_{1}(t) \\
C_{2}(t) \\
\vdots \\
C_{N-2}(t) \\
C_{N-1}(t) \\
\end{pmatrix}
\end{split}
$$
