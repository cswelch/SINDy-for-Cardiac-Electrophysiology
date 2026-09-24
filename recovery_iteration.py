import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from typing import cast

from scipy.integrate import odeint
from scipy.optimize import brentq
from gen_library_fit import GenLibraryFit
from fhn_models import fhn, fhn_c, fhn_vf_4, fhn_vf_7, compare_exact_and_sindy_coeffs

'''
    Proposed variant of partially observed SINDy where voltage is known but recovery is unknown.
    
    Given:  Voltage trace, stimulus (assume fixed period for now)
    Need:  Recovery variable v
    Methodology:  Guess a value for the recovery variable, generate the local action potential shape given that value, keep guessing values until that shape matches the data, then forecast
    the next value, etc.
'''

# Parameter that tells rest of script which model we want to test.
#   0 = FHN
#   1 = FHN-c
#   2 = VF4
#   3 = VF7
#   4 = FHN w/ pacedown
model_idx = 3

# Logical step function non-autonomous term; note that this must have ONLY one argument for PySINDy to handle it properly.
# Params:
#   t (1d array): Time input vector
#   period (float): The period of the stimulus
#   dur (float): The time duration of the stimulus
#   mag (float): The magnitude of the stimulus
def func_log(t):
    period=155.0
    dur=5.0
    mag=0.12

    stimulus = mag * (np.mod(t, period) <= dur)
    return stimulus

def func_log_vf_4(t):
    period=225.0
    dur=5.0
    mag=0.12

    stimulus = mag * (np.mod(t, period) <= dur)
    return stimulus

def func_log_vf_7(t):
    period=361.0
    dur=5.0
    mag=0.12

    stimulus = mag * (np.mod(t, period) <= dur)
    return stimulus

'''
    Function that defines stimulus with 3 sets of stim_count pulses, each with decreasing period.
    Params:
        t (array): Time values
        dur (float): Duration of each pulse
        mag (float): Magnitude of the stimulus
        stim_count (int): Number of stimuli of each period to generate
        periods (array): Array of periods for each set of pulses
'''
def func_pacedown(t):
    dur=5.0
    mag=0.12
    stim_count=3
    periods = np.array([305., 230., 155.])
    
    period_0_multiple = np.minimum(np.floor(t / periods[0]), stim_count)
    period_1_multiple = np.minimum(np.floor((t - period_0_multiple * periods[0]) / periods[1]), stim_count)
    period_2_multiple = np.minimum(np.floor((t - period_0_multiple * periods[0] - period_1_multiple * periods[1]) / periods[2]), stim_count)

    local_t_0 = t
    local_t_1 = t - stim_count * periods[0]
    local_t_2 = t - stim_count * periods[0] - stim_count * periods[1]

    is_stim = np.zeros_like(t, dtype=bool)

    mask_0 = period_0_multiple < stim_count
    mask_1 = (~mask_0) & (period_1_multiple < stim_count)
    mask_2 = (~mask_0) & (~mask_1) & (period_2_multiple < stim_count)

    stim_0 = np.mod(local_t_0, periods[0]) <= dur
    stim_1 = np.mod(local_t_1, periods[1]) <= dur
    stim_2 = np.mod(local_t_2, periods[2]) <= dur

    # Use np.where to allow both scalar and array t to work.
    is_stim = np.where(
        mask_0,
        stim_0,
        np.where(mask_1,
                 stim_1,
                 np.where(mask_2,
                          stim_2,
                          False))
    )

    stimulus = mag * is_stim
    return stimulus

dt = 0.05    # Time step
root_find_rel_tol = 1e-3 # Relative tolerance for Brent's method application (starts to fail around 0.5–1)
t_end_sim = 2000    # Upper bound of integration
t_end_vis = 2000     # Upper bound of x-axis on plots
n = int(t_end_sim / dt)   # Number of time steps
t_fhn = np.arange(0, t_end_sim, dt)    # Time range for integration

# Models
models = [fhn,
          fhn_c,
          fhn_vf_4,
          fhn_vf_7,
          fhn]

# Their corresponding variant string names
model_names = ['standard',
               'cardiac',
               'vf4',
               'vf7',
               'standard']

# ICs
x_0_fhn = np.array([-0.1,0])
x_0_fhn_c = np.array([0, 0.11])
x_0_vf_4 = np.array([0, 0.11])
x_0_vf_7 = np.array([0, 0])
ics = [x_0_fhn,
       x_0_fhn_c,
       x_0_vf_4,
       x_0_vf_7,
       x_0_fhn]

# Stimulus functions
funcs = [func_log,
         func_log,
         func_log_vf_4,
         func_log_vf_7,
         func_pacedown]

# Plot colors
colors = ['teal',
          'red',
          'blue',
          'green',
          'purple']

states_fhn = odeint(models[model_idx], ics[model_idx], t_fhn, args=(funcs[model_idx],), hmax=0.1) # Real n x 2 reference matrix of [u, v]

# Start w/ initial recovery variable value
v_old_est = 0
estimated_vs = []
for i in range(n-1):
    t_old = dt * i
    t_new = dt * (i+1)

    u_old = states_fhn[i, 0]
    u_new = states_fhn[i+1, 0]

    # Infer the recovery value by solving for the value that reproduces u_new.
    def voltage_error(v_candidate):
        ics_cur = np.array([u_old, v_candidate])
        t_cur = np.array([t_old, t_new])
        out = odeint(models[model_idx], ics_cur, t_cur, args=(funcs[model_idx],), hmax=0.1)
        return out[1, 0] - u_new

    lower = v_old_est - 0.05
    upper = v_old_est + 0.05
    while voltage_error(lower) * voltage_error(upper) > 0:
        lower -= 0.05
        upper += 0.05

    v_old_est = cast(float, brentq(voltage_error, lower, upper, rtol=root_find_rel_tol))

    estimated_vs.append(v_old_est)

print('Length of t_fhn: ', len(t_fhn))
print('Length of estimated_vs: ', len(estimated_vs))

plt.figure()
plt.plot(t_fhn, states_fhn[:, 0])
plt.title('Measured Voltages')
plt.xlabel('t (arbitrary units)')
plt.xlim(0, t_end_vis)
plt.ylabel('u (V)')

plt.figure()
plt.plot(t_fhn[:-1], estimated_vs, label='Estimated v')
plt.plot(t_fhn, states_fhn[:, 1], label='True v', linestyle=':')
plt.title('Estimated vs. True Recovery Variable Values')
plt.xlabel('t (arbitrary units)')
plt.xlim(0, t_end_vis)
plt.ylabel('v (recovery units)')
plt.legend()
plt.show()

# ------------------------------ Do the SINDy fit ------------------------------
print('--------- Starting SINDy fit with estimated recovery variable... ---------')
t_fit = t_fhn[:-1]
u_fit = states_fhn[:-1, 0]
v_estimated = np.asarray(estimated_vs)

assert len(t_fit) == len(u_fit) == len(v_estimated)

gen_library_fhn = GenLibraryFit(
    funcs[model_idx],
    funcs[model_idx],
    fhn_variant=model_names[model_idx],
    t_range=t_fit,
    ics=ics[model_idx],
    color=colors[model_idx],
    optimizer=ps.SSR(alpha=0.5, normalize_columns=False) # ps.STLSQ(threshold=0.001, normalize_columns=True)
)

# Replace simulated recovery data with estimated recovery data.
gen_library_fhn.t_fhn_td = t_fit
gen_library_fhn.dt = t_fit[1] - t_fit[0]
gen_library_fhn.states_fhn_td = np.column_stack(
    (u_fit, v_estimated, t_fit)
)

gen_library_fhn.fit(end_time=t_end_vis)
print('--------- SINDy fit finished. ---------')
plt.show()