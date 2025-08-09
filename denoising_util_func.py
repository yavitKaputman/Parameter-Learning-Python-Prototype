import numpy as np
import discrete_operators as D
import scipy as sp

# prox_G functions:
# TV:
def TV_prox_G(g, tau, v, u_hat):
    u = (u_hat + tau*(g + D.div(v))) / (1 + tau)
    return u

# TL:
def TL_prox_G(g, tau, v, u_hat):
    u = (u_hat + tau*(g - D.delta(v))) / (1 + tau)
    return u

# BH:
def BH_prox_G(g, tau, v, u_hat):
    u = (u_hat + tau*(g - D.div2(v))) / (1 + tau)
    return u

# Global:
def prox_G(regularizer_type, g, tau, v, u_hat):
    if regularizer_type == 'TV':
        u = TV_prox_G(g, tau, v, u_hat)
    elif regularizer_type == 'TL':
        u = TL_prox_G(g, tau, v, u_hat)
    elif regularizer_type == 'BH':
        u = BH_prox_G(g, tau, v, u_hat)
    else:
        print("Error, regularizer type must be TV or TL or BH!!!")
    return u

# prox_F functions:
# TV:
def TV_prox_F(alpha, mu, u_bar, v):
    v_argument = v + mu*(D.grad(u_bar))
    v_argument_0 = v_argument[..., 0]
    v_argument_1 = v_argument[..., 1]
    v_argument_norm = np.sqrt(v_argument_0**2 + v_argument_1**2)
    coefficient_matrix = np.maximum(alpha, v_argument_norm)
    coefficient_matrix = alpha / coefficient_matrix
    new_v = np.zeros(v.shape)
    new_v[..., 0] = coefficient_matrix * v_argument_0
    new_v[..., 1] = coefficient_matrix * v_argument_1
    return new_v

# TL:
def TL_prox_F(alpha, mu, u_bar, v):
    v_argument = v + mu*(D.delta(u_bar))
    v_argument_norm = np.sqrt(np.sum(v_argument))
    coefficient = np.maximum(alpha, v_argument_norm)
    coefficient = alpha / coefficient
    new_v = coefficient * v_argument
    return new_v

# BH:
def BH_prox_F(alpha, mu, u_bar, v):
    v_argument = v + mu*(D.hessian(u_bar))
    v_argument_0 = v_argument[..., 0]
    v_argument_1 = v_argument[..., 1]
    v_argument_2 = v_argument[..., 2]
    v_argument_3 = v_argument[..., 3]
    v_argument_norm = np.sqrt(v_argument_0**2 + v_argument_1**2
                              + v_argument_2**2 + v_argument_3**2)
    coefficient_matrix = np.maximum(alpha, v_argument_norm)
    coefficient_matrix = alpha / coefficient_matrix
    new_v = np.zeros(v.shape)
    new_v[..., 0] = coefficient_matrix * v_argument_0
    new_v[..., 1] = coefficient_matrix * v_argument_1
    new_v[..., 2] = coefficient_matrix * v_argument_2
    new_v[..., 3] = coefficient_matrix * v_argument_3
    return new_v

# Global:
def prox_F(regularizer_type, alpha, mu, u_bar, v):
    if regularizer_type == 'TV':
        u = TV_prox_F(alpha, mu, u_bar, v)
    elif regularizer_type == 'TL':
        u = TL_prox_F(alpha, mu, u_bar, v)
    elif regularizer_type == 'BH':
        u = BH_prox_F(alpha, mu, u_bar, v)
    else:
        print("Error, regularizer type must be TV or TL or BH!!!")
    return u

# Duality gap functions:
# TV:
def TV_Gap(g, alpha, u, v):
    div_v = D.div(v)
    TV_u = D.TV(u)
    gap_squared_normed_argument_1 = np.sum((u - g) ** 2)
    gap_squared_normed_argument_2 = np.sum(div_v ** 2)
    gap = (0.5)*(gap_squared_normed_argument_1) + alpha*TV_u + (0.5)*(gap_squared_normed_argument_2) + np.sum(div_v * g)
    return gap

# TL:
def TL_Gap(g, alpha, u, v):
    delta_v = D.delta(v)
    TL_u = D.TL(u)
    gap_squared_normed_argument_1 = np.sum((u - g) ** 2)
    gap_squared_normed_argument_2 = np.sum(delta_v ** 2)
    gap = (0.5)*(gap_squared_normed_argument_1) + alpha*TL_u + (0.5)*(gap_squared_normed_argument_2) - np.sum(delta_v * g)
    return gap

# BH:
def BH_Gap(g, alpha, u, v):
    div2_v = D.div2(v)
    BH_u = D.BH(u)
    gap_squared_normed_argument_1 = np.sum((u - g) ** 2)
    gap_squared_normed_argument_2 = np.sum(div2_v ** 2)
    gap = (0.5)*(gap_squared_normed_argument_1) + alpha*BH_u + (0.5)*(gap_squared_normed_argument_2) - np.sum(div2_v * g)
    return gap

# Global:
def Gap(regularizer_type, g, alpha, u, v):
    if regularizer_type == 'TV':
        gap = TV_Gap(g, alpha, u, v)
    elif regularizer_type == 'TL':
        gap = TL_Gap(g, alpha, u, v)
    elif regularizer_type == 'BH':
        gap = BH_Gap(g, alpha, u, v)
    else:
        print("Error, regularizer type must be TV or TL or BH!!!")
    return gap

def initial_v(regularizer_type, m, n):
    if regularizer_type == 'TV':
        v = np.zeros((m, n, 2))
    elif regularizer_type == 'TL':
        v = np.zeros((m, n))
    elif regularizer_type == 'BH':
        v = np.zeros((m, n, 4))
    else:
        print("Error, regularizer type must be TV or TL or BH!!!")
    return v
