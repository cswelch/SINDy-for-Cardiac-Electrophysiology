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

# Logical step function non-autonomous term; note that this must have ONLY one argument for PySINDy to handle it properly.
# Params:
#   t (1d array): Time input vector
#   period (float): The period of the stimulus
#   dur (float): The time duration of the stimulus
#   mag (float): The magnitude of the stimulus
def stimulus(t):
    period=155.0
    dur=5.0
    mag=0.12

    stimulus = mag * (np.mod(t, period) <= dur)
    return stimulus

dt = 1e-1   # Time step
t_end = 2000 # Upper bound of integration
n = int(t_end / dt)   # Number of time steps
t_fhn = np.arange(0, t_end, dt)    # Time range for integration
x_0_fhn = np.array([0, 0])   # ICs
states_fhn = odeint(fhn, x_0_fhn, t_fhn, args=(stimulus,), hmax=0.01) # Real n x 2 reference matrix of [u, v]

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
        out = odeint(fhn, ics_cur, t_cur, args=(stimulus,), hmax=0.01)
        return out[1, 0] - u_new

    lower = v_old_est - 0.05
    upper = v_old_est + 0.05
    while voltage_error(lower) * voltage_error(upper) > 0:
        lower -= 0.05
        upper += 0.05

    v_old_est = cast(float, brentq(voltage_error, lower, upper, xtol=1e-8))

    estimated_vs.append(v_old_est)

print('Length of t_fhn: ', len(t_fhn))
print('Length of estimated_vs: ', len(estimated_vs))

plt.figure()
plt.plot(t_fhn, states_fhn[:, 0])
plt.title('Measured Voltages')
plt.xlabel('t (arbitrary units)')
plt.ylabel('u (V)')

plt.figure()
plt.plot(t_fhn[:-1], estimated_vs, label='Estimated v')
plt.plot(t_fhn, states_fhn[:, 1], label='True v', linestyle=':')
plt.title('Estimated vs. True Recovery Variable Values')
plt.xlabel('t (arbitrary units)')
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
    stimulus,
    stimulus,
    fhn_variant="standard",
    t_range=t_fit,
    ics=x_0_fhn,
    color="blue",
    optimizer=ps.SSR(alpha=0.5, normalize_columns=False) # ps.STLSQ(threshold=0.001, normalize_columns=True)
)

# Replace simulated recovery data with estimated recovery data.
gen_library_fhn.t_fhn_td = t_fit
gen_library_fhn.dt = t_fit[1] - t_fit[0]
gen_library_fhn.states_fhn_td = np.column_stack(
    (u_fit, v_estimated, t_fit)
)

gen_library_fhn.fit(end_time=t_end)
print('--------- SINDy fit finished. ---------')
plt.show()