import numpy as np
import pysindy as ps
from scipy.integrate import odeint
from fhn_models import fhn, fhn_c, fhn_vf_b, compare_exact_and_sindy_coeffs
import matplotlib.pyplot as plt

class GenLibraryFit():
    '''
    Define class to create a GeneralizedLibrary and fit FHN with a specified non-autonomous term. Initialize the class with a 
    non-autonomous term function.
    Params:
        func (function): A function that takes time as input and returns a non-autonomous term.
        dt (float): The time step for generating data.
        non_aut_term_data (function): A function that takes time as input and returns a non-autonomous term for data generation.
        non_aut_term_fit (function): A function that takes time as input and returns a non-autonomous term for fitting.
        fhn_variant (string): The variant of the FitzHugh-Nagumo equations to use - "standard" for FHN, "cardiac" for FHN-c, and "vf" for VF-b variant.
    '''
    def __init__(self, dt, non_aut_term_data, non_aut_term_fit, fhn_variant="standard"):
        # Generate data w/ standard FHN parameters
        self.dt = dt
        self.t_fhn_td = np.arange(0,2000,dt)
        self.x_0_fhn_td = np.array([-0.1,0])    # TODO adjust v_0; may want a bit lower

        self.non_aut_term_data = non_aut_term_data
        self.non_aut_term_fit = non_aut_term_fit
        
        if fhn_variant == "standard":
            self.fhn_name = "standard"
            self.fhn_variant = self.fhn_td
        elif fhn_variant == "cardiac":
            self.fhn_name = "cardiac"
            self.fhn_variant = self.fhn_c_td
        elif fhn_variant == "vf":
            self.fhn_name = "vf"
            self.fhn_variant = self.fhn_vf_b_td

        # Generate u, v, and t data. Concatenate the t terms instead of directly solving for them since they are trivial.
        self.states_fhn_td = odeint(self.fhn_variant, self.x_0_fhn_td, self.t_fhn_td, hmax=0.1) # Test --> lambda t: 0       Actual --> logical_non_aut
        # plt.plot(t_fhn_td, states_fhn_td[:, 0], label='u')
        self.states_fhn_td = np.concatenate((self.states_fhn_td, self.t_fhn_td.reshape(-1, 1)), axis=1)

        # print(t_fhn_td.shape)
        # print(states_fhn_td.shape)
        # print(states_fhn_td[0:10,:])

    '''
    FitzHugh-Nagumo equations modified with an additional equation for time.
    Params:
        state (2d array):        Contains the state variables u and v
        t (1d array):            Time input
        non_aut_term (1d array): A non-autonomous term to be added to the u_dot equation
        alpha, beta, gamma, delta, eps, [theta, mu] (float): Parameters for the FHN equations.
    '''
    # Define standard FHN w/ non_aut_term_data term
    def fhn_td(self, state, t):
        return fhn(state, t, self.non_aut_term_data)
    
    # Define cardiac FHN w/ non_aut_term_data term
    def fhn_c_td(self, state, t):
        return fhn_c(state, t, self.non_aut_term_data)
    
    # Define VF-b variant of FHN w/ non_aut_term_data term
    def fhn_vf_b_td(self, state, t):
        return fhn_vf_b(state, t, self.non_aut_term_data)

    '''
    Fit the model using a GeneralizedLibrary with variable-specific libraries for u, v, and t.
    Params:
        non_aut_func (function): A function that takes time as input and returns a non-autonomous term.
    '''
    def fit(self):
        # Define variable-specific functions; functions of u and v are included in the first library, while the non-autonomous term is included in the second library. 
        # Note that the non-autonomous term is only applied to the u_dot equation.
        u_v_functions = [
            lambda u: u, 
            lambda u: u**2, 
            lambda u: u**3,
            lambda u, v: u*v
        ]

        t_functions = [
            lambda t: 1.0,
            self.non_aut_term_fit
        ]

        '''
        Create variable-specific libraries
        '''
        u_v_library = ps.CustomLibrary(library_functions=u_v_functions,
                                            function_names=[lambda u: u, lambda u: u + '^2', lambda u: u + '^3', lambda u, v: u + '*' + v])
        t_library = ps.CustomLibrary(library_functions=t_functions,
                                            function_names=[lambda t: 1, lambda t: 'f_td(' + t + ')'])
        
        # TODO Specify that t terms can only be in the u_dot and t_dot equations.
        '''
        Specify which functions apply to which variables. The ith row of inputs_per_library corresponds to the ith library, 
        and the jth column corresponds to the jth feature. Duplicate values are needed to ensure we don't have a ragged
        array with differing numbers of entries in each row. E.g., [[0,0,0], [1,1,2], [0,1,2]] would mean that the first
        library is applied to the first feature (u), the second library is applied to the second and third features (v,t),
        and the third library is applied to all three features (u,v,t). In our case, we have the following correspondence:
            u_v_library: Contains only functions of features 0 and 1 (u,v)
            t_library: Contains only functions of feature 2 (t)
        At the moment, we don't specify in which equations (e.g., u_dot, v_dot) each variable can be used.
        '''
        inputs_per_library = np.array([[0,1,1], 
                                       [2,2,2]])
        gen_library = ps.GeneralizedLibrary([u_v_library, t_library], inputs_per_library=inputs_per_library)


        # Fit SINDy model
        out = gen_library.fit(self.states_fhn_td)
        out = gen_library.transform(self.states_fhn_td)

        # Do the SINDy fit
        model_fhn_td = ps.SINDy(feature_names=["u", "v", "t"], feature_library=gen_library, optimizer=ps.SSR(alpha=1e-5, normalize_columns=True)) # ps.STLSQ(threshold=0.01, alpha=1e-5, normalize_columns=True)
        model_fhn_td.fit(self.states_fhn_td, t=self.t_fhn_td)

        # Create bar chart comparison between SINDy and exact coefficients.
        compare_exact_and_sindy_coeffs(model_fhn_td, self.fhn_name, non_aut_term_data=self.non_aut_term_data, non_aut_term_fit=self.non_aut_term_fit)