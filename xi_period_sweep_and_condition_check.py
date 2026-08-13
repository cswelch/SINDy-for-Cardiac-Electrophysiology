import numpy as np
from gen_library_fit import GenLibraryFit


def logical_non_aut(t, period=155.0, dur=5.0, mag=0.12):
    return mag * (np.mod(t, period) <= dur)


for period in [150.0, 155.0, 185.0, 225.0, 361.0]:
    def func(t, period=period):
            return logical_non_aut(t, period=period, dur=5.0, mag=0.12)
    
    for is_cubic in [True, False]:
        g = GenLibraryFit(
            func,
            func,
            fhn_variant='standard',
            t_range=np.arange(0, 500, 0.05),
            ics=np.array([-0.1, 0]),
            color='teal',
            tau=4.0,
        )
        try:
            g.fit_takens(printing=False, is_cubic=is_cubic, n_embed=3)
            mae = g.reconstruct_and_plot_takens(plotting=False, printing=False, end_time=500)
            print(f'period={period:>6.1f} cubic={is_cubic} MAE={mae:.3e}')
        except Exception as exc:
            print(f'period={period:>6.1f} cubic={is_cubic} ERROR {type(exc).__name__}: {exc}')

print('\nConditioning sweep:')
for period in [150.0, 155.0, 185.0, 225.0, 361.0]:
    def func_cur(t, period=period):
        return logical_non_aut(t, period=period, dur=5.0, mag=0.12)

    g = GenLibraryFit(
        func_cur,
        func_cur,
        fhn_variant='standard',
        t_range=np.arange(0, 500, 0.05),
        ics=np.array([-0.1, 0]),
        color='teal',
        tau=4.0,
    )
    u = g.states_fhn_td[:, 0]
    for tau_val in [2.0, 4.0, 8.0, 16.0, 25.0, 32.0, 64.0, 128.0]:
        delay_idx = int(round(tau_val / g.dt))
        for n_embed in [2, 3, 5]:
            total_delay = (n_embed - 1) * delay_idx
            n_samples = len(u) - total_delay
            if n_samples <= 0:
                continue
            X = np.zeros((n_samples, n_embed))
            for i in range(n_embed):
                idx = total_delay - i * delay_idx
                X[:, i] = u[idx:idx + n_samples]
            cond = np.linalg.cond(X)
            print(f'period={period:>6.1f} tau={tau_val:>4.1f} n={n_embed} cond={cond:.3e}')