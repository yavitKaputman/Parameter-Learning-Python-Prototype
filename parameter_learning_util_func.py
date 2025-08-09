import numpy as np
import discrete_operators as D
import scipy as sp

def g_bar(g_set, n, N):
    g_bar_set = np.ones((n+1, N))
    for i in range(N):
        g = g_set[...,i]
        vectorized_g = g.flatten(order='F')
        g_bar_set[0:n, i] = vectorized_g
    return g_bar_set

def A_Tilde(A, c_set, g_bar_set, Lambda, N):
    argument_matrix = A - ((1 / (Lambda * N)) * (g_bar_set * c_set) @ g_bar_set.T)
    eigenvalues, eigenvectors = np.linalg.eigh(argument_matrix)
    new_eigenvalues = np.maximum(0, eigenvalues)
    A_tilde = eigenvectors @ np.diag(new_eigenvalues) @ eigenvectors.T
    return A_tilde

# v_tilde functions:
# TV:
def TV_v_tilde(frequent_argument_set, g_bar_set, A_tilde, p, N):
    v_tilde_set = np.zeros((p, p, 2, N))
    for i in range(N):
        g_bar_i = g_bar_set[:, i]
        bound_i = g_bar_i.T @ A_tilde @ g_bar_i
        v_tilde_i = frequent_argument_set[..., i]
        coefficient_matrix = np.sqrt(v_tilde_i[..., 0] ** 2 + v_tilde_i[..., 1] ** 2)
        zero_indices = coefficient_matrix == 0
        coefficient_matrix[zero_indices] = bound_i
        coefficient_matrix = bound_i / coefficient_matrix
        v_tilde_set[..., 0, i] = coefficient_matrix * v_tilde_i[..., 0]
        v_tilde_set[..., 1, i] = coefficient_matrix * v_tilde_i[..., 1]
    return v_tilde_set

# TL:
def TL_v_tilde(frequent_argument_set, g_bar_set, A_tilde, p, N):
    v_tilde_set = np.zeros((p, p, N))
    for i in range(N):
        g_bar_i = g_bar_set[:, i]
        bound_i = g_bar_i.T @ A_tilde @ g_bar_i
        v_tilde_i = frequent_argument_set[..., i]
        coefficient = np.sqrt(np.sum(v_tilde_i ** 2))
        if coefficient == 0:
            coefficient = bound_i
        coefficient = bound_i / coefficient
        v_tilde_set[..., i] = coefficient * v_tilde_i
    return v_tilde_set

# BH:
def BH_v_tilde(frequent_argument_set, g_bar_set, A_tilde, p, N):
    v_tilde_set = np.zeros((p, p, 4, N))
    for i in range(N):
        g_bar_i = g_bar_set[:, i]
        bound_i = g_bar_i.T @ A_tilde @ g_bar_i
        v_tilde_i = frequent_argument_set[..., i]
        coefficient_matrix = np.sqrt(v_tilde_i[..., 0] ** 2 + v_tilde_i[..., 1] ** 2
                                     + v_tilde_i[..., 2] ** 2 + v_tilde_i[..., 3] ** 2)
        zero_indices = coefficient_matrix == 0
        coefficient_matrix[zero_indices] = bound_i
        coefficient_matrix = bound_i / coefficient_matrix
        v_tilde_set[..., 0, i] = coefficient_matrix * v_tilde_i[..., 0]
        v_tilde_set[..., 1, i] = coefficient_matrix * v_tilde_i[..., 1]
        v_tilde_set[..., 2, i] = coefficient_matrix * v_tilde_i[..., 2]
        v_tilde_set[..., 3, i] = coefficient_matrix * v_tilde_i[..., 3]
    return v_tilde_set

def v_tilde(regularizer_type, frequent_argument_set, g_bar_set, A_tilde, p, N):
    if regularizer_type == 'TV':
        v_tilde_set = TV_v_tilde(frequent_argument_set, g_bar_set, A_tilde, p, N)
    elif regularizer_type == 'TL':
        v_tilde_set = TL_v_tilde(frequent_argument_set, g_bar_set, A_tilde, p, N)
    elif regularizer_type == 'BH':
        v_tilde_set = BH_v_tilde(frequent_argument_set, g_bar_set, A_tilde, p, N)
    else:
        print("Error, regularizer type must be TV or TL or BH!!!")
    return v_tilde_set

# Regularized u_plus functions:
# TV:
def TV_u_plus(u_plus_set, N):
    TV_u_plus_set = np.zeros(N)
    for i in range(N):
        TV_u_plus_set[i] = D.TV(u_plus_set[..., i])
    return TV_u_plus_set

# TL:
def TL_u_plus(u_plus_set, N):
    TL_u_plus_set = np.zeros(N)
    for i in range(N):
        TL_u_plus_set[i] = D.TL(u_plus_set[..., i])
    return TL_u_plus_set

# BH:
def BH_u_plus(u_plus_set, N):
    BH_u_plus_set = np.zeros(N)
    for i in range(N):
        BH_u_plus_set[i] = D.BH(u_plus_set[..., i])
    return BH_u_plus_set

def regularized_u_plus(regularizer_type, u_plus_set, N):
    if regularizer_type == 'TV':
        regularized_u_plus_set = TV_u_plus(u_plus_set, N)
    elif regularizer_type == 'TL':
        regularized_u_plus_set = TL_u_plus(u_plus_set, N)
    elif regularizer_type == 'BH':
        regularized_u_plus_set = BH_u_plus(u_plus_set, N)
    else:
        print("Error, regularizer type must be TV or TL or BH!!!")
    return regularized_u_plus_set

# D_f functions:
# TV:
def TV_D_f(A_direction_squared_norm, v_direction_set, N):
    D_f_Av = A_direction_squared_norm
    for i in range(N):
        v_direction_i = v_direction_set[..., i]
        D_f_Av += (np.sum(v_direction_i[...,0] ** 2 + v_direction_i[...,1] ** 2))
    return D_f_Av

# TL:
def TL_D_f(A_direction_squared_norm, v_direction_set, N):
    D_f_Av = A_direction_squared_norm
    for i in range(N):
        v_direction_i = v_direction_set[..., i]
        D_f_Av += (np.sum(v_direction_i ** 2))
    return D_f_Av

# BH:
def BH_D_f(A_direction_squared_norm, v_direction_set, N):
    D_f_Av = A_direction_squared_norm
    for i in range(N):
        v_direction_i = v_direction_set[..., i]
        D_f_Av += (np.sum(v_direction_i[...,0] ** 2 + v_direction_i[...,1] ** 2 + v_direction_i[...,2] ** 2 + v_direction_i[...,3] ** 2))
    return D_f_Av

# Global:
def D_f(regularizer_type, A_direction_squared_norm, v_direction_set, N):
    if regularizer_type == 'TV':
        D_f_Av = TV_D_f(A_direction_squared_norm, v_direction_set, N)
    elif regularizer_type == 'TL':
        D_f_Av = TL_D_f(A_direction_squared_norm, v_direction_set, N)
    elif regularizer_type == 'BH':
        D_f_Av = BH_D_f(A_direction_squared_norm, v_direction_set, N)
    else:
        print("Error, regularizer type must be TV or TL or BH!!!")
    return D_f_Av

def Theta(regularizer_type, A_direction, v_direction_set, N, res, Lambda):
    A_direction_squared_norm = np.sum(A_direction ** 2)
    D_f_Av = D_f(regularizer_type, A_direction_squared_norm, v_direction_set, N)
    step_size = (N * (res + ((Lambda / 2) * A_direction_squared_norm))) / (8 * D_f_Av)
    theta = np.minimum(1, step_size)
    return theta

# Iterative calculations functions:
# TV:
def TV_iterative_calculations(regularized_u_plus_set, g_set, v_set, p, N):
    frequent_argument_set = np.zeros((p, p, 2, N))
    c_set = np.zeros(N)
    for i in range(N):
        v_i = v_set[..., i]
        g_i = g_set[..., i]
        frequent_argument_set[..., i] = D.grad_div(v_i) + D.grad(g_i)
        frequent_argument_i = frequent_argument_set[..., i]
        beta = - np.sum(np.sqrt(frequent_argument_i[..., 0] ** 2 + frequent_argument_i[..., 1] ** 2))
        c_set[i] = regularized_u_plus_set[i] + beta
    return frequent_argument_set, c_set

# TL:
def TL_iterative_calculations(regularized_u_plus_set, g_set, v_set, p, N):
    frequent_argument_set = np.zeros((p, p, N))
    c_set = np.zeros(N)
    for i in range(N):
        v_i = v_set[..., i]
        g_i = g_set[..., i]
        frequent_argument_set[..., i] = D.delta_delta(v_i) - D.delta(g_i)
        frequent_argument_i = frequent_argument_set[..., i]
        beta = np.sqrt(np.sum(frequent_argument_i ** 2))
        c_set[i] = regularized_u_plus_set[i] + beta
    return frequent_argument_set, c_set

# BH:
def BH_iterative_calculations(regularized_u_plus_set, g_set, v_set, p, N):
    frequent_argument_set = np.zeros((p, p, 4, N))
    c_set = np.zeros(N)
    for i in range(N):
        v_i = v_set[..., i]
        g_i = g_set[..., i]
        frequent_argument_set[..., i] = D.hessian_div2(v_i) - D.hessian(g_i)
        frequent_argument_i = frequent_argument_set[..., i]
        beta = np.sum(np.sqrt(frequent_argument_i[..., 0] ** 2 + frequent_argument_i[..., 1] ** 2
                              + frequent_argument_i[..., 2] ** 2 + frequent_argument_i[..., 3] ** 2))
        c_set[i] = regularized_u_plus_set[i] + beta
    return frequent_argument_set, c_set

# Global:
def iterative_calculations(regularizer_type, regularized_u_plus_set, g_set, v_set, p, N):
    if regularizer_type == 'TV':
        frequent_argument_set, c_set = TV_iterative_calculations(regularized_u_plus_set, g_set, v_set, p, N)
    elif regularizer_type == 'TL':
        frequent_argument_set, c_set = TL_iterative_calculations(regularized_u_plus_set, g_set, v_set, p, N)
    elif regularizer_type == 'BH':
        frequent_argument_set, c_set = BH_iterative_calculations(regularized_u_plus_set, g_set, v_set, p, N)
    else:
        print("Error, regularizer type must be TV or TL or BH!!!")
    return frequent_argument_set, c_set

# Residual functions:
# TV:
def TV_Res(regularized_u_plus_set, frequent_argument_set, g_bar_set, A_direction, v_direction_set, Lambda, N):
    coefficient = 1 / N
    grad_f_A = coefficient * ((g_bar_set * regularized_u_plus_set) @ g_bar_set.T)
    res = np.sum(grad_f_A * A_direction) - ((Lambda / 2) * ((np.sum(A_direction)) ** 2))
    grad_f_v = 0
    for i in range(N):
        grad_f_v_i = -frequent_argument_set[..., i]
        v_direction_i = v_direction_set[..., i]
        grad_f_v += np.sum((grad_f_v_i[..., 0] * v_direction_i[..., 0]) + (grad_f_v_i[..., 1] * v_direction_i[..., 1]))
    grad_f_v *= coefficient
    res += grad_f_v
    return res

# TL:
def TL_Res(regularized_u_plus_set, frequent_argument_set, g_bar_set, A_direction, v_direction_set, Lambda, N):
    coefficient = 1 / N
    grad_f_A = coefficient * ((g_bar_set * regularized_u_plus_set) @ g_bar_set.T)
    res = np.sum(grad_f_A * A_direction) - ((Lambda / 2) * ((np.sum(A_direction)) ** 2))
    grad_f_v = 0
    for i in range(N):
        grad_f_v_i = frequent_argument_set[..., i]
        v_direction_i = v_direction_set[..., i]
        grad_f_v += np.sum(grad_f_v_i * v_direction_i)
    grad_f_v *= coefficient
    res += grad_f_v
    return res

# BH:
def BH_Res(regularized_u_plus_set, frequent_argument_set, g_bar_set, A_direction, v_direction_set, Lambda, N):
    coefficient = 1 / N
    grad_f_A = coefficient * ((g_bar_set * regularized_u_plus_set) @ g_bar_set.T)
    res = np.sum(grad_f_A * A_direction) - ((Lambda / 2) * ((np.sum(A_direction)) ** 2))
    grad_f_v = 0
    for i in range(N):
        grad_f_v_i = frequent_argument_set[..., i]
        v_direction_i = v_direction_set[..., i]
        grad_f_v += np.sum((grad_f_v_i[..., 0] * v_direction_i[..., 0]) + (grad_f_v_i[..., 1] * v_direction_i[..., 1])
                           + (grad_f_v_i[..., 2] * v_direction_i[..., 2]) + (grad_f_v_i[..., 3] * v_direction_i[..., 3]))
    grad_f_v *= coefficient
    res += grad_f_v
    return res

# Global:
def Res(regularizer_type, regularized_u_plus_set, frequent_argument_set, g_bar_set, A_direction, v_direction_set, Lambda, N):
    if regularizer_type == 'TV':
        res = TV_Res(regularized_u_plus_set, frequent_argument_set, g_bar_set, A_direction, v_direction_set, Lambda, N)
    elif regularizer_type == 'TL':
        res = TL_Res(regularized_u_plus_set, frequent_argument_set, g_bar_set, A_direction, v_direction_set, Lambda, N)
    elif regularizer_type == 'BH':
        res = BH_Res(regularized_u_plus_set, frequent_argument_set, g_bar_set, A_direction, v_direction_set, Lambda, N)
    else:
        print("Error, regularizer type must be TV or TL or BH!!!")
    return res

def save_trained_data(regularizer_type, A, v_set, Iterations, Residuals):
    sp.io.savemat(f'data/output/trained data/{regularizer_type}_trained_data.mat', {
        f'{regularizer_type}_A': A,
        f'{regularizer_type}_v_Set': v_set,
        f'{regularizer_type}_Iterations': Iterations,
        f'{regularizer_type}_Residuals': Residuals,
    })




