import numpy as np
from gen_library_fit import GenLibraryFit


def logical_non_aut(t, period=155.0, dur=5.0, mag=0.12):
    return mag * (np.mod(t, period) <= dur)


for period in [155.0, 185.0, 225.0]:
    def func(t, period=period):
        return logical_non_aut(t, period=period, dur=5.0, mag=0.12)
    
    for is_cubic in [False, True]:
        try:
            g = GenLibraryFit(
                func,
                func,
                fhn_variant='standard',
                t_range=np.arange(0, 500, 0.05),
                ics=np.array([-0.1, 0]),
                tau=4.0,
            )
            g.fit_takens(printing=False, is_cubic=is_cubic, n_embed=3)
            mae = g.reconstruct_and_plot_takens(plotting=False, printing=False, end_time=500)

            X = g.takens_X_embedded[:, :-1]
            Theta = g.takens_model.feature_library.transform(g.takens_X_embedded)
            cond_x = np.linalg.cond(X)
            cond_theta = np.linalg.cond(Theta)
            coef = g.takens_model.coefficients()
            max_coef = np.max(np.abs(coef))
            nz = np.sum(np.abs(coef) > 1e-8)
            print(
                f"period={period:>6.1f} cubic={is_cubic} MAE={mae:.3e} "
                f"condX={cond_x:.3e} condTheta={cond_theta:.3e} "
                f"max|coef|={max_coef:.3e} nz={nz}"
            )
        except Exception as exc:
            print(f"period={period:>6.1f} cubic={is_cubic} ERROR {type(exc).__name__}: {exc}")
