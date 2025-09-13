import numpy as np
import pysindy as ps
from scipy.integrate import odeint
from fhn_models import fhn, fhn_c, fhn_vf_4, fhn_vf_7, compare_exact_and_sindy_coeffs
import matplotlib.pyplot as plt

class GenLibraryFit():
    '''
    Define class to create a GeneralizedLibrary and fit FHN with a specified non-autonomous term. Initialize the class with a 
    non-autonomous term function.
    Params:
        non_aut_term_data (function): A function that takes time as input and returns a non-autonomous term for data generation.
        non_aut_term_fit (function): A function that takes time as input and returns a non-autonomous term for fitting.
        fhn_variant (string): The variant of the FitzHugh-Nagumo equations to use; "standard" for FHN, "cardiac" for FHN-c, and "vf" for VF-b variant.
        t_range (1d array): The time range over which to simulate the FHN equations, including start time, end time, and time step dt.
        ics (1d array): Initial conditions for the FHN equations.
        color (string): Color to use for the original data in the reconstruction plots.
        u_noise (float): Standard deviation of Gaussian noise to add to the u variable. Default is 0.0 (no noise).
        v_noise (float): Standard deviation of Gaussian noise to add to the v variable. Default is 0.0 (no noise).
    '''
    def __init__(self, non_aut_term_data, non_aut_term_fit, fhn_variant="standard", t_range=np.arange(0,2000,0.01), ics=np.array([-0.1,0]), color="blue", u_noise=0.0, v_noise=0.0):
        # Generate data w/ standard FHN parameters
        self.t_fhn_td = t_range
        self.x_0_fhn_td = ics
        self.color = color

        self.non_aut_term_data = non_aut_term_data
        self.non_aut_term_fit = non_aut_term_fit
        
        if fhn_variant == "standard":
            self.fhn_name = "standard"
            self.fhn_variant = self.fhn_td
        elif fhn_variant == "cardiac":
            self.fhn_name = "cardiac"
            self.fhn_variant = self.fhn_c_td
        elif fhn_variant == "VF4":
            self.fhn_name = "VF4"
            self.fhn_variant = self.fhn_vf_4_td
        elif fhn_variant == "VF7":
            self.fhn_name = "VF7"
            self.fhn_variant = self.fhn_vf_7_td

        # Generate u, v, and t data. Concatenate the t terms instead of directly solving for them since they are trivial.
        self.states_fhn_td = odeint(self.fhn_variant, self.x_0_fhn_td, self.t_fhn_td, hmax=0.1) # Test --> lambda t: 0       Actual --> logical_non_aut

        # int_u = np.trapz(y=self.states_fhn_td[:, 0], dx=0.01)
        # print(f"Average u value: {int_u / 4000.}")
        # int_v = np.trapz(y=self.states_fhn_td[:, 1], dx=0.01)
        # print(f"Average v value: {int_v / 4000.}")

        # Add Gaussian noise with the given std dev if nonzero values are provided.
        if u_noise > 0.0:
            self.states_fhn_td[:, 0] += np.random.normal(0, u_noise, self.states_fhn_td[:, 0].shape) # Add noise to u
        if v_noise > 0.0:
            self.states_fhn_td[:, 1] += np.random.normal(0, v_noise, self.states_fhn_td[:, 1].shape) # Add noise to v
        
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
    

    # Define VF-a (VF-4) variant of FHN w/ non_aut_term_data term
    def fhn_vf_4_td(self, state, t):
        return fhn_vf_4(state, t, self.non_aut_term_data)
    

    # Define VF-b (VF-7) variant of FHN w/ non_aut_term_data term
    def fhn_vf_7_td(self, state, t):
        return fhn_vf_7(state, t, self.non_aut_term_data)


    '''
        Reconstruct the system using the fitted SINDy model and plot the results.
        Params:
            model (pysindy.SINDy): A fitted SINDy model.
            t (1d array): Time range for simulation.
            x_0 (1d array): Initial conditions for simulation.
    '''
    def reconstruct_and_plot(self, model, t, x_0):
        # Make the initial condition match the training data (2 components (u_0,v_0) --> 3 components (u_0,v_0,t_0))
        x_0 = np.concatenate((x_0, np.array([t[0]])))
        model_reconstruction = model.simulate(x_0, t, integrator='odeint')

        fig, ax = plt.subplots(2, 1, figsize=(8, 8), dpi=200)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3) # Add space between the two plots

        # Plot for u and v variables
        for i in range(model_reconstruction.shape[1] - 1):
            ax[i].plot(self.states_fhn_td[:, 2], self.states_fhn_td[:, i], label='Exact Solution', color=self.color)
            ax[i].plot(t, model_reconstruction[:,i], label='SINDy Reconstruction', color='black', linestyle='--')
            ax[i].set_xlim(0, t[-1])
            ax[i].set_xlabel('t')
            ax[i].set_ylabel(['u', 'v'][i])
            ax[i].set_title([f'Voltage vs. Time ({self.fhn_name}, {self.non_aut_term_data.__name__})', f'Recovery Variable vs. Time ({self.fhn_name}, {self.non_aut_term_data.__name__})'][i])
            # ax[i].grid()
            # if (i == 0):
            #     ax[i].set_ylim(-0.05, 0.9) # Set constanty limits for u plot to maintain comparability
            if (i == 1):
                # ax[i].set_ylim(0.08, 0.185) # Set constant limits for v plot to maintain comparability
                ax[i].legend() # Only add legend to the 2nd plot to save space
            


    '''
    Fit the model using a GeneralizedLibrary with variable-specific libraries for u, v, and t.
    Params:
        non_aut_func (function): A function that takes time as input and returns a non-autonomous term.
    Returns:
        model_fhn_td (pysindy.SINDy): A fitted SINDy model with the specified non-autonomous term.
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

        # TODO Getting f_td in both u' and v' equations for cases with large dt and/or noise. How to fix?
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


        # TODO Is there any use for fit / transform here?
        # out = gen_library.fit(self.states_fhn_td)
        # out = gen_library.transform(self.states_fhn_td)

        # Do the SINDy fit
        model_fhn_td = ps.SINDy(feature_names=["u", "v", "t"], feature_library=gen_library, optimizer=ps.SSR(alpha=1e-5, normalize_columns=True)) # ps.STLSQ(threshold=0.01, alpha=1e-5, normalize_columns=True)
        model_fhn_td.fit(self.states_fhn_td, t=self.t_fhn_td)

        # Create bar chart comparison between SINDy and exact coefficients.
        compare_exact_and_sindy_coeffs(model_fhn_td, self.fhn_name, non_aut_term_data=self.non_aut_term_data, non_aut_term_fit=self.non_aut_term_fit)

        # Reconstruct the solution from the SINDy fit and plot it against the data.        
        self.reconstruct_and_plot(model_fhn_td, self.t_fhn_td, self.x_0_fhn_td)

        return model_fhn_td