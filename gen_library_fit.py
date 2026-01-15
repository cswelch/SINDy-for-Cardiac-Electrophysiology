import numpy as np
import pysindy as ps
from scipy.integrate import odeint
from ddeint import ddeint
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
    def __init__(self, non_aut_term_data, non_aut_term_fit, fhn_variant="standard", 
            t_range=np.arange(0,2000,0.01), ics=np.array([-0.1,0]), color="blue", 
            u_noise=0.0, v_noise=0.0, tau=None):
        
        # Initialize with Takens embedding using 5 time delays.
        self.t_fhn_td = t_range
        self.x_0_fhn_td = ics
        self.color = color
        self.non_aut_term_data = non_aut_term_data
        self.non_aut_term_fit = non_aut_term_fit
        
        # Compute dt for delay calculations
        self.dt = t_range[1] - t_range[0] if len(t_range) > 1 else 1.0
        
        # Auto-compute optimal delay if not provided
        if tau is None:
            self.tau = self._compute_delay(fhn_variant, ics, t_range[:500], 
                                        non_aut_term_data)
        else:
            self.tau = tau
                
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
        elif fhn_variant == "standard_delayed_copy":
            self.tau = 24 # Set the delay time for the delayed copy variant
            self.fhn_name = "standard_delayed_copy"
            self.fhn_variant = self.fhn_delayed_copy
        elif fhn_variant == "fhn_auto_osc_delayed_copy":
            self.tau = 24 # Set the delay time for the delayed copy variant
            self.fhn_name = "fhn_auto_osc_delayed_copy"
            self.fhn_variant = self.fhn_auto_osc_delayed_copy

        # Function wrapper for constant ICs passed to ddeint so they are callable as it expects.
        def initial_history(t):
            return self.x_0_fhn_td

        # Generate u, v, and t data. Concatenate the t terms instead of directly solving for them since they are trivial.
        if fhn_variant == "standard_delayed_copy":
            # Wraps additional arguments sent to delayed function
            fhn_variant_func = lambda Y, t: GenLibraryFit.fhn_delayed_copy(Y, t, self.non_aut_term_data, self.tau)
            # For any delayed copy variant, use ddeint instead
            self.states_fhn_td = ddeint(fhn_variant_func, initial_history, self.t_fhn_td)
        elif fhn_variant == "fhn_auto_osc_delayed_copy":  # For non-delayed variants, use odeint
            # Wraps additional arguments sent to delayed function
            fhn_variant_func = lambda Y, t: GenLibraryFit.fhn_auto_osc_delayed_copy(Y, t, self.tau)
            self.states_fhn_td = ddeint(fhn_variant_func, initial_history, self.t_fhn_td)
        else:
            # For non-delayed variants (i.e., "standard" + "cardiac" + "VF4" + "VF7"), use odeint
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
        Estimate optimal time delay using Average Mutual Information (AMI).
    '''
    def _compute_delay(self, fhn_variant, ics, t_short, non_aut_term):
        # Generate short trajectory for analysis.
        if fhn_variant == "standard":
            fhn_func = lambda Y, t: fhn(Y, t, non_aut_term)
        else:
            fhn_func = lambda Y, t: fhn(Y, t, non_aut_term) # TODO Add additional variants if needed.
        
        u_short = odeint(fhn_func, ics, t_short)[:, 0]
        
        # Compute mutual information at different lags.
        max_lag = len(u_short) // 10
        ami = np.zeros(max_lag)
        
        for lag in range(1, max_lag):
            # Simple entropy-based mutual information proxy.
            u1 = u_short[:-lag]
            u2 = u_short[lag:]
            
            # Find first minimum in AMI curve (first zero crossing).
            correlation = np.corrcoef(u1, u2)[0, 1]
            ami[lag] = abs(correlation)
        
        # First minimum gives good delay (where AMI first drops significantly).
        first_min_idx = np.argmax(np.gradient(np.gradient(ami[:max_lag//2])))
        optimal_delay = max(1, first_min_idx)
        
        # Convert from steps to time.
        return int(optimal_delay) * self.dt


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
        Define versions with v' defined as delayed version of u' equation. Done as FHN system to be
        passed to ddeint where Y(t) gives current values and Y(t-tau) gives delayed values.
    '''
    def fhn_delayed_copy(Y, t, non_aut_term, tau, alpha=0.1):
        u, v = Y(t)
        
        # Original u equation
        u_dot = u*(1-u)*(u-alpha) - v + non_aut_term(t)
        
        # For v equation, we need the delayed u_dot value
        # This requires reconstructing u_dot at time (t-tau)
        if t >= tau:
            u_delayed, v_delayed = Y(t - tau)
            u_dot_delayed = u_delayed*(1-u_delayed)*(u_delayed-alpha) - v_delayed + non_aut_term(t - tau)
        else:
            u_dot_delayed = 0  # Set u' to 0 if in t range (0, tau)
        
        v_dot = u_dot_delayed
        
        return np.array([u_dot, v_dot])
        # return fhn_u_dot_copy(state, t, self.non_aut_term_data)

    '''
        Define delayed copy variant for auto-oscillatory case of FHN.
    '''
    # TODO Finish fhn_auto_osc_delayed_copy.
    # Define versions with v' defined as delayed version of u' equation
    def fhn_auto_osc_delayed_copy(Y, t, tau, alpha=0.1):
        u, v = Y(t)
        
        # Original u equation
        u_dot = u*(1-u)*(u-alpha) - v
        
        # For v equation, we need the delayed u_dot value
        # This requires reconstructing u_dot at time (t-tau)
        if t >= tau:
            u_delayed, v_delayed = Y(t - tau)
            u_dot_delayed = u_delayed*(1-u_delayed)*(u_delayed-alpha) - v_delayed
        else:
            u_dot_delayed = 0  # Set u' to 0 if in t range (0, tau)
        
        v_dot = u_dot_delayed
        
        return np.array([u_dot, v_dot])

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

        fig, ax = plt.subplots(2, 1, figsize=(8, 8)) # For presentations and papers, use:  figsize=(8, 8), dpi=200
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3) # Add space between the two plots

        # Plot for u and v variables
        for i in range(model_reconstruction.shape[1] - 1):
            ax[i].plot(self.states_fhn_td[:, 2], self.states_fhn_td[:, i], label='Exact Solution', color=self.color)
            ax[i].plot(t, model_reconstruction[:,i], label='SINDy Reconstruction', color='black', linestyle='--')
            ax[i].set_xlim(0, 400) # t[-1]
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

        # Do the SINDy fit
        model_fhn_td = ps.SINDy(feature_names=["u", "v", "t"], feature_library=gen_library, optimizer=ps.SSR(alpha=1e-5, normalize_columns=True)) # ps.STLSQ(threshold=0.01, alpha=1e-5, normalize_columns=True)
        model_fhn_td.fit(self.states_fhn_td, t=self.t_fhn_td)

        # Create bar chart comparison between SINDy and exact coefficients.
        compare_exact_and_sindy_coeffs(model_fhn_td, self.fhn_name, non_aut_term_data=self.non_aut_term_data, non_aut_term_fit=self.non_aut_term_fit)

        # Reconstruct the solution from the SINDy fit and plot it against the data.        
        self.reconstruct_and_plot(model_fhn_td, self.t_fhn_td, self.x_0_fhn_td)

        return model_fhn_td


    '''
    Fit SINDy using Takens time-delay embedding with 5 dimensions.
    Reconstructs the 2D phase space from a single measured variable u.
    '''
    # TODO Figure out how to compare output with single-shift ID results.
    def fit_takens(self):
        # Constrain ourselves to extract only u and t since v wouldn't be observable experimentally.
        u = self.states_fhn_td[:, 0]
        t = self.t_fhn_td
        
        # Compute delay indices.
        delay_idx = int(np.round(self.tau / self.dt))
        delay_idx = max(1, min(delay_idx, len(u) // 6))  # Sanity check
        
        # Define our 5-dimensional embedding by [u(t), u(t-τ), u(t-2τ), u(t-3τ), u(t-4τ)].
        n_embed = 5
        total_delay = (n_embed - 1) * delay_idx
        
        # Initialize embedded state matrix.
        n_samples = len(u) - total_delay
        X_embedded = np.zeros((n_samples, n_embed + 1))  # +1 for time
        
        for i in range(n_embed):
            idx = total_delay - i * delay_idx
            X_embedded[:, i] = u[idx:idx + n_samples]
        
        X_embedded[:, n_embed] = t[total_delay:total_delay + n_samples]
        
        feature_names = [f"u(t-{i*delay_idx})" for i in range(n_embed)] + ["t"]
        
        # --- Build polynomial library for 5D embedding with up to degree 3 for u terms. ---
        embed_library = ps.PolynomialLibrary(degree=3)
        
        # --- Build library of time-dependent terms. ---
        t_functions = [
            lambda t: 1.0,
            self.non_aut_term_fit
        ]
        t_library = ps.CustomLibrary(
            library_functions=t_functions,
            function_names=[lambda t: 1, lambda t: 'f_td(' + t + ')']
        )
        
        # Combine libraries with GeneralizedLibrary.
        inputs_per_library = np.array([
            [0, 1, 2, 3, 4], # Apply embedding library to first 5 features (u dimensions)
            [5, 5, 5, 5, 5]  # Apply time library to last feature (t)
        ])
        gen_library = ps.GeneralizedLibrary(
            [embed_library, t_library],
            inputs_per_library=inputs_per_library
        )
        
        # Fit SINDy model.
        model = ps.SINDy(
            feature_names=feature_names,
            feature_library=gen_library,
            optimizer=ps.SSR(alpha=1e-5, normalize_columns=True)
        )
        
        model.fit(X_embedded, t=t[total_delay:total_delay + n_samples])
        
        print("\n--- SINDy with Takens Embedding (5 dimensions) ---")
        print(f"Optimal time delay τ = {self.tau:.4f} (delay_idx = {delay_idx})")
        print(f"Embedding dimension = {n_embed}")
        model.print()
        
        return model
    

    '''
        Fit SINDy using only the u variable by embedding into (u, u_dot) space.
        Mathematically, FHN can be written as a 2nd order ODE for u.
    '''
    def fit_latent_ODE(self):
        # Constrain ourselves to extract only u and t since v wouldn't be observable experimentally.
        u_obs = self.states_fhn_td[:, 0]
        t = self.t_fhn_td
        
        # Compute u_dot numerically to form the embedded state X = [u, u_dot] using PySINDy's built-in differentiation method.
        diff_method = ps.FiniteDifference() 
        u_dot_obs = diff_method._differentiate(u_obs, t=t)
        
        # Create new state [u, u_dot], appending t to the end for the library generation.
        X_embedded = np.stack([u_obs, u_dot_obs], axis=1)
        X_with_time = np.concatenate((X_embedded, t.reshape(-1, 1)), axis=1)
        
        # TODO Still missing terms from L-ODE formulation — both I(t) and I'(t) needed?
        # --- Build library for u only (index 0) ---
        u_only_functions = [
            lambda u: u,
            lambda u: u**2,
            lambda u: u**3
        ]
        u_only_library = ps.CustomLibrary(
            library_functions=u_only_functions,
            function_names=[
                lambda u: u,
                lambda u: u + '^2',
                lambda u: u + '^3'
            ]
        )

        # --- Build library for u_dot only (index 1) ---
        u_dot_only_functions = [
            lambda u_dot: u_dot
        ]
        u_dot_only_library = ps.CustomLibrary(
            library_functions=u_dot_only_functions,
            function_names=[
                lambda u_dot: u_dot
            ]
        )

        # --- Build library for u and u_dot (indices 0 and 1, respectively) ---
        u_and_u_dot_functions = [
            lambda u, u_dot: u * u_dot,
            lambda u, u_dot: u**2 * u_dot
        ]
        u_and_u_dot_library = ps.CustomLibrary(
            library_functions=u_and_u_dot_functions,
            function_names=[
                lambda u, u_dot: u + '*' + u_dot,
                lambda u, u_dot: u + '^2*' + u_dot
            ]
        )

        # --- Build library for time (index 2) to include forcing terms ---
        t_functions = [
            lambda t: 1.0,
            self.non_aut_term_fit
        ]
        t_library = ps.CustomLibrary(
            library_functions=t_functions,
            function_names=[lambda t: 1, lambda t: 'f_td(' + t + ')']
        )

        # Map libraries to inputs as follows:
        inputs_per_library = np.array([
            [0, 0, 0], # u_only_library: u corresponds to input 0
            [1, 1, 1], # u_dot_only_library: u_dot corresponds to input 1
            [0, 1, 1], # u_and_u_dot_library: (u, u_dot) correspond to inputs 0 and 1, respectively
            [2, 2, 2]  # t_library: t corresponds to input 2
        ])
        
        gen_library = ps.GeneralizedLibrary(
            [u_only_library, u_dot_only_library, u_and_u_dot_library, t_library], 
            inputs_per_library=inputs_per_library
        )

        # Fit SINDy model.
        #    Target: \dot{X} = [\dot{u}, \ddot{u}]. 
        #    SINDy learns: u' = u_dot (trivial), u'' = f(...) (nontrivial)
        optimizer = ps.SSR(alpha=1e-5, normalize_columns=True)
        model_latent = ps.SINDy(
            feature_names=["u", "u_dot", "t"], 
            feature_library=gen_library, 
            optimizer=optimizer
        )

        model_latent.fit(X_with_time, t=t) # Pass X_with_time (3 cols); t handles implicit time column usage.

        # Debugging — Print number of features for each library and shape of the input data
        # print(f"Number of features in u_only library: {u_only_library.n_output_features_}")
        # print(f"Number of features in u_and_u_dot library: {u_and_u_dot_library.n_output_features_}")
        # print(f"Number of features in time library: {t_library.n_output_features_}")
        # print(f"Number of features in generalized library: {gen_library.n_output_features_}")
        # print(X_with_time.shape)
        # print(t.shape)

        print("Identified Latent Model (u, u_dot):")
        model_latent.print()
        
        return model_latent