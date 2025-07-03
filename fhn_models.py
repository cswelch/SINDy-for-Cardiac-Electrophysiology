import numpy as np
import matplotlib.pyplot as plt

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
def fhn_c(state, t, non_aut_term, alpha = 0.1, beta = 1, gamma = 1.6, delta = 0.0, eps = 0.01):
    u, v = state
    u_dot = u*(1-u)*(u-alpha) - u*v   # Note - The cardiac version without hyperpolarization (FHN-c) has -u*v instead of -v. ID success can change depending on which variant we use.
    v_dot = eps*(beta*u - gamma*v - delta)

    # Add time-dependent voltage perturbation
    u_dot +=  non_aut_term(t)
    return u_dot, v_dot

'''
Generate the data using the VF-b modified FHN equations:
Returns:
    u_dot (float): The time derivative of the state variable u.
    v_dot (float): The time derivative of the state variable v.
Params:
    state (2d array):        Contains the state variables u and v
    t (1d array):            Time input
    non_aut_term (1d array): A non-autonomous term to be added to the u_dot equation
    alpha = 0.2, beta = 1.1, gamma = 0.31, delta = 1.0, eps = 0.005, theta = -0.05, mu = 1.0 (Old auto-oscillatory parameters used to get these equations)
'''
def fhn_vf_b(state, t, non_aut_term, alpha = 0.2, beta = 1.1, gamma = 0.31, delta = 0.0, eps = 0.005, theta = -0.05, mu = 1.0):
    u, v = state
    u_dot = mu*u*(1 - u)*(u - alpha) - u*v  # Note - The cardiac version without hyperpolarization (FHN-c) has -u*v instead of -v. ID success can change depending on which variant we use.
    v_dot = eps*((beta - u)*(u - gamma) - delta*v - theta)

    # Add time-dependent voltage perturbation
    u_dot +=  non_aut_term(t)
    return u_dot, v_dot

def compare_exact_and_sindy_coefs(variables: list, coef_sindy: np.ndarray, coef_exact: np.ndarray):
    # Order of coefficients: 1, u, v, u^2, uv, v^2, u^3, u^2v, uv^2, v^3
    variables = ["1", "u", "v", "u^2", "uv", "v^2", "u^3", "u^2v", "uv^2", "v^3"]
    coef_sindy = model_vf_b.coefficients()
    coef_exact = np.array([[0, -0.22, 0, 0.42, -1.0, 0, -0.2, 0, 0, 0],
                        [0.0515, 0.3193, -0.00309, -1.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    x = np.arange(len(coef_sindy[0]))

    # Use bar chart to compare the SINDy and exact coefficient values.
    width = 0.4
    plt.figure(figsize=(8, 4))
    plt.bar(x - width/2, coef_sindy[0], width=width, alpha=0.9, label='SINDy')
    plt.bar(x + width/2, coef_exact[0], width=width, alpha=0.9, label='Exact')

    plt.title('Exact vs. SINDy Coefficients for FHN VF-b')
    plt.ylabel('Coefficient Values')
    plt.xlabel('Variables')
    plt.xticks(x, variables)
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)