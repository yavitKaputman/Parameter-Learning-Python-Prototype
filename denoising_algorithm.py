import numpy as np
import denoising_util_func as DUF

def RCP(regularizer_type, tau, mu, max_It, least_Res, stat_update_iter, g, alpha):
    m, n = g.shape
    u = np.zeros((m, n))
    u_bar = np.copy(u)
    v = DUF.initial_v(regularizer_type, m, n)
    iteration = 0
    gap = 1
    while gap >= least_Res and iteration <= max_It:
        iteration += 1
        old_gap = np.copy(gap)
        u_hat = np.copy(u)
        v = DUF.prox_F(regularizer_type, alpha, mu, u_bar, v)
        u = DUF.prox_G(regularizer_type, g, tau, v, u_hat)
        u_bar = 2*u - u_hat
        gap = DUF.Gap(regularizer_type, g, alpha, u, v)
        #if iteration % stat_update_iter == 0:
        #    print(f"Iteration {iteration}, Duality Gap: {gap:.6f}")
        if gap < 0:
            u = u_hat
            gap = old_gap
            break
    return u, gap