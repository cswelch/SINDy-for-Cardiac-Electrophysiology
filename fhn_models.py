import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
from typing import Callable
import sympy as sym
from sklearn import metrics

'''
Standard FitzHugh-Nagumo equations for comparison w/ the above method.
Returns:
    u_dot (float): The time derivative of the state variable u.
    v_dot (float): The time derivative of the state variable v.
Params:
    state (2d array):        Contains the state variables u and v
    t (1d array):            Time input
    non_aut_term (1d array): A non-autonomous term to be added to the u_dot equation
    alpha = 0.2, beta = 1.1, gamma = 0.31, delta = 0.0, eps = 0.005 (use delta = 1.0 for auto-oscillatory state)
'''
def fhn(state, t, non_aut_term, alpha = 0.1, beta = 0.5, gamma = 1, delta = 0.0, eps = 0.01):
    u, v = state
    u_dot = u*(1-u)*(u-alpha) - v   # Note - The cardiac version without hyperpolarization (FHN-c) has -u*v instead of -v. ID success can change depending on which variant we use.
    v_dot = eps*(beta*u - gamma*v - delta)

    # Add time-dependent voltage perturbation
    u_dot +=  non_aut_term(t)
    return u_dot, v_dot


'''
FHN-c (cardiac variant of FHN) system definition.
Returns:
    u_dot (float): The time derivative of the state variable u.
    v_dot (float): The time derivative of the state variable v.
Params:
    state (2d array):        Contains the state variables u and v
    t (1d array):            Time input
    non_aut_term (1d array): A non-autonomous term to be added to the u_dot equation
    alpha = 0.1, beta = 1, gamma = 1.6, delta = 0.0, eps = 0.01 (use delta = 1.0 for auto-oscillatory state)
'''
def fhn_c(state, t, non_aut_term, alpha = 0.1, beta = 0.5, gamma = 1, delta = 0.0, eps = 0.01):
    u, v = state
    u_dot = u*(1-u)*(u-alpha) - u*v   # Note - The cardiac version without hyperpolarization (FHN-c) has -u*v instead of -v. ID success can change depending on which variant we use.
    v_dot = eps*(beta*u - gamma*v - delta)

    # Add time-dependent voltage perturbation
    u_dot +=  non_aut_term(t)
    return u_dot, v_dot


'''
Generate the data using the 4-parameter VF-a modified FHN equations from Velasco et al.
Returns:
    u_dot (float): The time derivative of the state variable u.
    v_dot (float): The time derivative of the state variable v.
Params:
    state (2d array):        Contains the state variables u and v
    t (1d array):            Time input
    non_aut_term (1d array): A non-autonomous term to be added to the u_dot equation
    alpha = 0.2, beta = 1.1, eps = 0.005, mu = 1.0: Default parameters for the VF-a equations.
'''
def fhn_vf_4(state, t, non_aut_term, alpha = 0.2, beta = 1.1, eps = 0.005, mu = 1.0):
    u, v = state
    u_dot = mu * u * (1-u) * (u-alpha) - u*v  # Note - The cardiac version without hyperpolarization (FHN-c) has -u*v instead of -v. ID success can change depending on which variant we use.
    v_dot = eps * (u * (beta-u) - v)

    # Add time-dependent voltage perturbation
    u_dot +=  non_aut_term(t)
    return u_dot, v_dot


'''
Generate the data using the 7-parameter VF-b modified FHN equations from Velasco et al.
Returns:
    u_dot (float): The time derivative of the state variable u.
    v_dot (float): The time derivative of the state variable v.
Params:
    state (2d array):        Contains the state variables u and v
    t (1d array):            Time input
    non_aut_term (1d array): A non-autonomous term to be added to the u_dot equation
    alpha = 0.2, beta = 1.1, gamma = 0.31, delta = 1.0, eps = 0.005, theta = -0.05, mu = 1.0 (Old auto-oscillatory parameters used to get these equations)
'''
def fhn_vf_7(state, t, non_aut_term, alpha = 0.2, beta = 1.1, gamma = 0.31, delta = 0.0, eps = 0.005, theta = -0.05, mu = 1.0):
    u, v = state
    u_dot = mu*u*(1 - u)*(u - alpha) - u*v  # Note - The cardiac version without hyperpolarization (FHN-c) has -u*v instead of -v. ID success can change depending on which variant we use.
    v_dot = eps*((beta - u)*(u - gamma) - delta*v - theta)

    # Add time-dependent voltage perturbation
    u_dot +=  non_aut_term(t)
    return u_dot, v_dot


# TODO Figure out how to make this handle 2nd-order L-ODE forms to compare exact coefficients automatically.
'''
Generate bar chart to compare the exact and fit SINDy coefficients for the given SINDy model. Also perform a checksum on the number of terms.
Returns:
    None
Params:
    model (ps.SINDy):   The SINDy model from which to extract the fit coefficients.
    fhn_name (str):     The fhn method (chosen from 1. "standard" 2. "cardiac" 3. "vf") used to evaluate the exact coefficients.
    precision (int):    Exponent with base 10 of tolerance below which terms are considered zero. (E.g., 5 for 1e-5.)
'''
def compare_exact_and_sindy_coeffs(model: ps.SINDy, fhn_name: str, non_aut_term_data: Callable, non_aut_term_fit: Callable, precision: int = 5):
    # Order of coefficients: 1, u, v, u^2, uv, v^2, u^3, u^2v, uv^2, v^3
    # Use SymPy to symbolically evaluate the FHN variant, summing like terms and extracting the coefficients.
    monomial_names = model.get_feature_names()
    # print(f"Monomials: {monomial_names}")
    coef_sindy = model.coefficients()
    coef_exact = get_fhn_exact_coeffs(fhn_name=fhn_name, monomial_names=monomial_names)

    # TODO Handle different u', v' names for L-ODE case?
    # Compare number of terms in the SINDy model to the exact coefficients.
    tol = 10**(-precision)
    print(f"Number of SINDy terms for u\', v\': ({np.sum(np.abs(coef_sindy[0]) > tol)}, {np.sum(np.abs(coef_sindy[1]) > tol)})")
    print(f"Number of exact terms for u\', v\': ({np.sum(np.abs(coef_exact[0]) > tol)}, {np.sum(np.abs(coef_exact[1]) > tol)})")
    mae_coef = metrics.mean_absolute_error(coef_exact, coef_sindy[0:2])
    print(f"Mean Absolute Error of coefficients: {mae_coef:.{precision}e}")

    # Compare the SINDy and exact coefficients.
    print("\nSINDy coefficients:")
    model.print(precision=precision)
    print("\nExact coefficients:")
    for i in range(len(coef_exact)):
        eqn = ""
        eqn += ("u\' = " if i == 0 else "v\' = ")
        # Return the index of the final nonzero coefficient in the current equation (or -1 if all coefficients are zero).
        final_nonzero_index = np.nonzero(np.abs(coef_exact[i]) > tol)[0][-1] if np.any(np.abs(coef_exact[i]) > tol) else -1
        
        for j in range(len(coef_exact[0])):
            if (np.abs(coef_exact[i][j]) > tol) and (j < final_nonzero_index):  # Print only nonzero coefficients.
                eqn += f"{coef_exact[i][j]:.{precision}f} {monomial_names[j]} + "
            elif (j == final_nonzero_index):  # Handle the last term separately.
                eqn += f"{coef_exact[i][j]:.{precision}f} {monomial_names[j]}"
        print(eqn)


    # Use bar chart to compare the SINDy and exact coefficient values (for u').
    width = 0.4
    x = np.arange(len(coef_sindy[0]))
    plt.figure(figsize=(8, 4))
    plt.bar(x - width/2, coef_sindy[0], width=width, alpha=0.9, label='SINDy')
    plt.bar(x + width/2, coef_exact[0], width=width, alpha=0.9, label='Exact')

    # Print the titles. Include the fhn_name and string form of the non-autonomous data method.
    plt.title(f'SINDy vs. Exact Coefficients for {fhn_name} with {non_aut_term_data.__name__} Data and {non_aut_term_fit.__name__} Fit (u\')')
    plt.ylabel('Coefficient Values')
    plt.xlabel('Monomials')
    plt.xticks(x, monomial_names)
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)


    # Use bar chart to compare the SINDy and exact coefficient values (for v').
    width = 0.4
    plt.figure(figsize=(8, 4))
    plt.bar(x - width/2, coef_sindy[1], width=width, alpha=0.9, label='SINDy')
    plt.bar(x + width/2, coef_exact[1], width=width, alpha=0.9, label='Exact')

    plt.title(f'SINDy vs. Exact Coefficients for {fhn_name} with {non_aut_term_data.__name__} Data and {non_aut_term_fit.__name__} Fit (v\')')
    plt.ylabel('Coefficient Values')
    plt.xlabel('Monomials')
    plt.xticks(x, monomial_names)
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)


'''
Get the exact coefficients for the FHN equations using SymPy.
Returns:
    (np.array): 2D array containing u' and v' coefficients, respectively.
Params:
    fhn_variant (str):              The variant of the FHN equations to use. Options are "standard", "cardiac", or "vf".
    monomial_names (list of str):   List of monomial names to use in the equations.
    params (dict):                  Optional dictionary of parameters to use for the FHN equations. If None, default parameters are used.
'''
def get_fhn_exact_coeffs(fhn_name="standard", monomial_names = ['u', 'u**2', 'u**3', 'v', '1', 'f_td'], params=None):
    # Define symbols
    u, v, f_td = sym.symbols('u v f_td')

    # Ensure monomial term names match those used in the equations.
    if 'f_td(t)' in monomial_names:
        monomial_names[monomial_names.index('f_td(t)')] = 'f_td'

    # print(f"Edited monomial names: {monomial_names}")

    # Default parameters
    if params is None:
        if fhn_name == "standard":
            params = dict(alpha = 0.1, beta = 0.5, gamma = 1, delta = 0.0, eps = 0.01)
        elif fhn_name == "cardiac":
            params = dict(alpha = 0.1, beta = 0.5, gamma = 1, delta = 0.0, eps = 0.01)
        elif fhn_name == "VF4":
            params = dict(alpha = 0.2, beta = 1.1, eps = 0.005, mu = 1.0)
        elif fhn_name == "VF7":
            params = dict(alpha = 0.2, beta = 1.1, gamma = 0.31, delta = 0.0, eps = 0.005, theta = -0.05, mu = 1.0)
        elif fhn_name == "fhn_lode":
            params = dict(alpha = 0.1, beta = 0.5, gamma = 1, delta = 0.0, eps = 0.01)
        else:
            raise ValueError("Unknown FHN variant")

    # TODO Would like to use preexisting functions from this module to avoid code duplication, but they aren't currently working with SymPy.
    # Build symbolic equations
    if fhn_name == "standard":
        u_rhs = u*(1-u)*(u-params['alpha']) - v + f_td
        v_rhs = params['eps']*(params['beta']*u - params['gamma']*v - params['delta'])
    elif fhn_name == "cardiac":
        u_rhs = u*(1-u)*(u-params['alpha']) - u*v + f_td
        v_rhs = params['eps']*(params['beta']*u - params['gamma']*v - params['delta'])
    elif fhn_name == "VF4":
        u_rhs = params['mu']*u*(1-u)*(u-params['alpha']) - u*v + f_td
        v_rhs = params['eps']*(u*(params['beta']-u) - v)
    elif fhn_name == "VF7":
        u_rhs = params['mu']*u*(1-u)*(u-params['alpha']) - u*v + f_td
        v_rhs = params['eps']*((params['beta']-u)*(u-params['gamma']) - params['delta']*v - params['theta'])
    elif fhn_name == "fhn_lode":
        # L-ODE formulation: eliminates v from standard FHN to get a 2nd-order ODE for u.
        # Returns coefficients for (u_dot, u_dot_dot) instead of (u_dot, v_dot).
        u_dot_sym = sym.Symbol('u_dot')
        f_td_dot = sym.Symbol('f_td_dot')

        # First eqn. (trivial): u_dot = u_dot
        u_dot_rhs = u_dot_sym

        # 2nd-order L-ODE eqn. (derived by eliminating v from standard FHN)
        u_dot_dot_rhs = (
            -3*u**2*u_dot_sym
            + 2*(params['alpha'] + 1)*u*u_dot_sym
            - (params['alpha'] + params['eps']*params['gamma'])*u_dot_sym
            - params['eps']*params['gamma']*u**3
            + params['eps']*params['gamma']*(params['alpha'] + 1)*u**2
            - params['eps']*(params['gamma']*params['alpha'] + params['beta'])*u
            + params['eps']*params['delta']
            + params['eps']*params['gamma']*f_td
            + f_td_dot
        )

        # Replace '^' with '**' in monomial names for compatibility w/ SymPy; fit_latent_ODE uses '^' notation
        monomial_names = [m.replace('^', '**') for m in monomial_names]

        # Expand and extract coefficients using (u, u_dot, f_td, f_td_dot) as generators instead of (u, v, f_td)
        u_dot_rhs_exp = sym.expand(u_dot_rhs).as_poly(u, u_dot_sym, f_td, f_td_dot)
        u_dot_dot_rhs_exp = sym.expand(u_dot_dot_rhs).as_poly(u, u_dot_sym, f_td, f_td_dot)

        m_sym = [sym.sympify(m) for m in monomial_names]

        u_dot_coeffs = [float(u_dot_rhs_exp.coeff_monomial(m)) for m in m_sym]
        u_dot_dot_coeffs = [float(u_dot_dot_rhs_exp.coeff_monomial(m)) for m in m_sym]

        return np.array([u_dot_coeffs, u_dot_dot_coeffs])
    else:
        raise ValueError("Unknown FHN variant")

    # Expand and collect coefficients
    u_rhs_exp = sym.expand(u_rhs).as_poly(u, v, f_td)
    v_rhs_exp = sym.expand(v_rhs).as_poly(u, v, f_td)

    m_sym = [sym.sympify(m) for m in monomial_names]

    u_coeffs = [float(u_rhs_exp.coeff_monomial(m)) for m in m_sym]
    v_coeffs = [float(v_rhs_exp.coeff_monomial(m)) for m in m_sym]
    # print("u_rhs_exp:", u_rhs_exp)
    # print("v_rhs_exp:", v_rhs_exp)
    # print("u_coeffs:", u_coeffs)
    # print("v_coeffs:", v_coeffs)

    return np.array([u_coeffs, v_coeffs])