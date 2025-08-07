import numpy as np
from numba import njit, prange

@njit
def reaction_rates(concs, rate_constants, reactants, products, nreactions):
    rates = np.zeros_like(concs)
    nreactions = rate_constants.shape[0]

    for r in range(nreactions):
        rate = rate_constants[r]
        for reactant in reactants[r]:
            if reactant == -1: break
            rate *= concs[reactant]

        for reactant in reactants[r]:
            if reactant == -1: break
            rates[reactant] -= rate
        for product  in products[r]: 
            if product == -1: break
            rates[product]  += rate

    return rates


@njit
def finite_difference_jacobian(y, rate_constants, reactants, products, nreactions, eps=1e-8):
    """
    Approximate Jacobian using finite differences.
    J[i, j] = d(dy_i/dt) / dy_j
    """
    n = len(y)
    J = np.zeros((n, n))
    f0 = reaction_rates(y, rate_constants, reactants, products, nreactions)

    for j in range(n):
        y_perturb = y.copy()
        y_perturb[j] += eps
        f1 = reaction_rates(y_perturb, rate_constants, reactants, products, nreactions)
        J[:, j] = (f1 - f0) / eps

    return J

@njit
def backward_euler_step(y0, dt, rate_constants, reactants, products, nreactions, max_iter=20, tol=1e-8):
    """
    Solve y_new = y0 + dt * f(y_new) using Newton-Raphson.
    """
    y = y0.copy()
    for _ in range(max_iter):
        f = reaction_rates(y, rate_constants, reactants, products, nreactions)
        g = y - y0 - dt * f

        if np.linalg.norm(g, ord=2) < tol:
            return y

        J = finite_difference_jacobian(y, rate_constants, reactants, products, nreactions)
        I = np.eye(len(y))
        A = I - dt * J
        delta = np.linalg.solve(A, -g)
        y += delta

        # Optional: clip to avoid negatives
        y = np.maximum(y, 0.0)

    raise RuntimeError("Backward Euler did not converge")

@njit(parallel=True)
def back_euler_parallel_update(concs_all, dt, rate_constants, reactants, products, nreactions, max_iter=3):
    nsolutes, npoints = concs_all.shape
    for point in prange(npoints):
        concs = concs_all[:, point].copy()
        concs_updated = backward_euler_step(concs, dt, rate_constants,
                                            reactants, products, nreactions,
                                            max_iter)
        concs_all[:, point] = concs_updated


class SolutionBackEulerMixin:
    def precompute_back_eulers(self):
        self.save_solute_to_index()

        self.rate_constants = []
        self.reactants = []
        self.products = []
        for reaction in self.reactions:
            self.rate_constants.append(reaction.rate_constant)
            self.reactants.append([
                self.solute_to_index[sol] for sol in reaction.reactants
            ])
            self.products.append([
                self.solute_to_index[sol] for sol in reaction.products
            ])
        max_reactants = max([len(r) for r in self.reactants])
        max_products = max([len(p) for p in self.products])

        def pad_indices(lst, maxlen): return lst + [-1] * (maxlen - len(lst))

        self.reactants = [pad_indices(r, max_reactants) for r in self.reactants]
        self.products = [pad_indices(p, max_products) for p in self.products]
        self.reactants = np.array(self.reactants)
        self.products = np.array(self.products)

        self.rate_constants = np.array(self.rate_constants)
        self.nreactions = len(self.reactions)
        self.nsolutes = len(self.solutes)

    
    def back_Euler_update(self, dt):
        # Shape: (nsolutes, npoints)
        concs_all = np.array([sol.conc for sol in self.solutes], dtype=np.float64)
        
        back_euler_parallel_update(concs_all, dt, self.rate_constants,
                                self.reactants, self.products,
                                self.nreactions)

        # Write updated concentrations back
        for sol_i, sol in enumerate(self.solutes):
            sol.conc[:] = concs_all[sol_i, :]
