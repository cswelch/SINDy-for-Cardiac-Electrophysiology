import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from gen_library_fit import GenLibraryFit
from fhn_models import fhn, fhn_c, fhn_vf_4, fhn_vf_7, compare_exact_and_sindy_coeffs

'''
    Proposed variant of partially observed SINDy where voltage is known but recovery is unknown.
    
    Given:  Voltage trace, stimulus (assume fixed period for now)
    Need:  Recovery variable v
    Methodology:  Guess a value for the recovery variable, generate the local action potential shape given that value, keep guessing values until that shape matches the data, then forecast
    the next value, etc.
'''

# Logical step function non-autonomous term.
# Params:
#   t (1d array): Time input vector
#   period (float): The period of the stimulus
#   dur (float): The time duration of the stimulus
#   mag (float): The magnitude of the stimulus
def logical_non_aut(t, period=155.0, dur=5.0, mag=0.12):
    stimulus = mag * (np.mod(t, period) <= dur)
    return stimulus

eps = 1e-2  # Error tolerance between estimated and actual voltage
inc = 1e-2  # Recovery increment / decrement amount
dt = 1.   # Time step
t_end = 200 # Upper bound of integration
n = int(t_end / dt)   # Number of time steps
t_fhn = np.arange(0, t_end, dt)    # Time range for integration
x_0_fhn = np.array([0, 0])   # ICs
states_fhn = odeint(fhn, x_0_fhn, t_fhn, args=(logical_non_aut,), hmax=0.01) # Real n x 2 reference matrix of [u, v]

# Start w/ initial recovery variable value
v_old_est = 0
estimated_vs = []
for i in range(n-1):
    t_old = dt * i
    t_new = dt * (i+1)

    u_old = states_fhn[i, 0]
    u_new = states_fhn[i+1, 0]

    # Assume we don't know the recovery variable (e.g., it can't be measured), then vary its value until u_new is sufficiently close to u_new_est
    ics_cur = np.array([u_old, v_old_est])
    t_cur = np.array([t_old, t_new])
    out = odeint(fhn, ics_cur, t_cur, args=(logical_non_aut,), hmax=dt)
    u_new_est, v_new_est = out[1, 0], out[1, 1]
    print(f't_old: {t_old}, t_new: {t_new}, u_old: {u_old}, u_new: {u_new}, u_new_est: {u_new_est}, v_old_est: {v_old_est}, v_new_est: {v_new_est}')
        
    while (abs(u_new_est - u_new) > eps):
        # Increase or decrease past recovery variable estimate depending on the direction the voltage needs to go
        if ((u_new_est - u_new) > 0):
            v_old_est += inc
            print(f'Incrementing to {v_old_est}')
        else:
            v_old_est -= inc
            print(f'Decrementing to {v_old_est}')

        ics_cur = np.array([u_old, v_old_est])
        t_cur = np.array([t_old, t_new])
        out = odeint(fhn, ics_cur, t_cur, args=(logical_non_aut,), hmax=dt)
        u_new_est, v_new_est = out[1, 0], out[1, 1]

        print(f't_old: {t_old}, t_new: {t_new}, u_old: {u_old}, u_new: {u_new}, u_new_est: {u_new_est}, v_old_est: {v_old_est}, v_new_est: {v_new_est}')

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
plt.plot(t_fhn, states_fhn[:, 1], label='True v')
plt.xlabel('t (arbitrary units)')
plt.ylabel('v (recovery units)')
plt.title('Estimated Recovery Variable vs. True Recovery Variable Values')
plt.legend()
plt.show()