import numpy as np

# First order operators:

def x_plus(u):
    m, n = u.shape
    Du = np.zeros((m, n))
    Du[:, 0:n - 1] = u[:, 1:n] - u[:, 0:n - 1]
    Du[:, n - 1] = u[:, 0] - u[:, n-1]
    return Du

def y_plus(u):
    m, n = u.shape
    Du = np.zeros((m, n))
    Du[0:m - 1, :] = u[1:m, :] - u[0:m - 1, :]
    Du[m - 1, :] = u[0, :] - u[m - 1, :]
    return Du

def x_minus(u):
    m, n = u.shape
    Du = np.zeros((m, n))
    Du[:, 1:n] = u[:, 1:n] - u[:, 0:n - 1]
    Du[:, 0] = u[:, 0] - u[:, n-1]
    return Du

def y_minus(u):
    m, n = u.shape
    Du = np.zeros((m, n))
    Du[1:m, :] = u[1:m, :] - u[0:m - 1, :]
    Du[0, :] = u[0, :] - u[m - 1, :]
    return Du

def grad(u):
    m, n = u.shape
    grad_u = np.zeros((m, n, 2))
    grad_u[..., 0] = x_plus(u)
    grad_u[..., 1] = y_plus(u)
    return grad_u

def TV(u):
    grad_u = grad(u)
    grad_u_0 = grad_u[..., 0]
    grad_u_1 = grad_u[..., 1]
    TV_u = np.sum(np.sqrt(grad_u_0**2 + grad_u_1**2))
    return TV_u

def div(v):
    div_v = x_minus(v[..., 0]) + y_minus(v[..., 1])
    return div_v

# Second order operators:

# x_plus x_minus = x_minus x_plus
def x_2_mixed_directions(u):
    m, n = u.shape
    Du = np.zeros((m, n))
    Du[:, 0] = u[:, n-1] - 2*u[:, 0] + u[:, 1]
    Du[:, 1:n-1] = u[:, 0:n-2] - 2*u[:, 1:n-1] + u[:, 2:n]
    Du[:, n-1] = u[:, n-2] - 2*u[:, n-1] + u[:, 0]
    return Du

# y_plus y_minus = y_minus y_plus
def y_2_mixed_directions(u):
    m, n = u.shape
    Du = np.zeros((m, n))
    Du[0, :] = u[m-1, :]- 2*u[0, :] + u[1, :]
    Du[1:m-1, :] = u[0:m-2, :]- 2*u[1:m-1, :] + u[2:m, :]
    Du[m-1, :] = u[m-2, :] - 2*u[m-1, :] + u[0, :]
    return Du

# x_plus y_plus = y_plus x_plus
def xy_plus(u):
    m, n = u.shape
    Du = np.zeros((m, n))
    Du[0:m-1, 0:n-1] = u[0:m-1, 0:n-1] - u[1:m, 0:n-1] - u[0:m-1, 1:n] + u[1:m, 1:n]
    Du[m-1, 0:n-1] = u[m-1, 0:n-1] - u[0, 0:n-1] - u[m-1, 1:n] + u[0, 1:n]
    Du[0:m-1, n-1] = u[0:m-1, n-1] - u[1:m, n-1]- u[0:m-1, 0] + u[1:m, 0]
    Du[m-1, n-1] = u[m-1, n-1] - u[0, n-1]- u[m-1, 0] + u[0, 0]
    return Du

# x_minus y_minus = y_minus x_minus
def xy_minus(u):
    m, n = u.shape
    Du = np.zeros((m, n))
    Du[0, 0] = u[0, 0] - u[0, n-1] - u[m-1, 0] + u[m-1, n-1]
    Du[0, 1:n] = u[0, 1:n] - u[0, 0:n-1] - u[m-1, 1:n] + u[m-1, 0:n-1]
    Du[1:m, 0] = u[1:m, 0] - u[1:m, n-1] - u[0:m-1, 0] + u[0:m-1, n-1]
    Du[1:m, 1:n] = u[1:m, 1:n] - u[1:m, 0:n-1] - u[0:m-1, 1:n] + u[0:m-1, 0:n-1]
    return Du

def x_plus_y_minus(u):
    m, n = u.shape
    Du = np.zeros((m, n))
    Du[0, 0:n-1] = u[0, 1:n] - u[0, 0:n-1] - u[m-1, 1:n] + u[m-1, 0:n-1]
    Du[0, n-1] = u[0, 0] - u[0, n-1]- u[m-1, 0] + u[m-1, n-1]
    Du[1:m, 0:n-1] = u[1:m, 1:n] - u[1:m, 0:n-1] - u[0:m-1, 1:n] + u[0:m-1, 0:n-1]
    Du[1:m, n-1] = u[1:m, 0] - u[1:m, n-1] - u[0:m-1, 0] + u[0:m-1, n-1]
    return Du

def y_plus_x_minus(u):
    m, n = u.shape
    Du = np.zeros((m, n))
    Du[0:m-1, 0] = u[1:m, 0] - u[0:m-1, 0] - u[1:m, n-1] + u[0:m-1, n-1]
    Du[m-1, 0] = u[0, 0] - u[m-1, 0] - u[0, n-1] + u[m-1, n-1]
    Du[0:m-1, 1:n] = u[1:m, 1:n] - u[0:m-1, 1:n] - u[1:m, 0:n-1] + u[0:m-1, 0:n-1]
    Du[m-1, 1:n] = u[0, 1:n] - u[m-1, 1:n] - u[0, 0:n-1] + u[m-1, 0:n-1]
    return Du

def grad_div(v):
    m, n, p = v.shape
    grad_div_v = np.zeros((m, n, p))
    grad_div_v[..., 0] = x_plus(x_minus(v[..., 0])) + x_plus(y_minus(v[..., 1]))
    grad_div_v[..., 1] = y_plus(x_minus(v[..., 0])) + x_plus(y_minus(v[..., 1]))

    grad_div_v[..., 0] = x_2_mixed_directions(v[..., 0]) + x_plus_y_minus(v[..., 1])
    grad_div_v[..., 1] = y_plus_x_minus(v[..., 0]) + y_2_mixed_directions(v[..., 1])
    return grad_div_v

def delta(u):
    delta_u = x_2_mixed_directions(u) + y_2_mixed_directions(u)
    return delta_u

def TL(u):
    delta_u = delta(u)
    TL_u = np.sqrt(np.sum(delta_u**2))
    return TL_u

def hessian(u):
    m, n = u.shape
    hessian_u = np.zeros((m, n, 4))
    hessian_u[..., 0] = x_2_mixed_directions(u)
    hessian_u[..., 1] = xy_plus(u)
    hessian_u[..., 2] = hessian_u[..., 1]
    hessian_u[..., 3] = y_2_mixed_directions(u)
    return hessian_u

def BH(u):
    hessian_u = hessian(u)
    hessian_u_0 = hessian_u[..., 0]
    hessian_u_1 = hessian_u[..., 1]
    hessian_u_2 = hessian_u[..., 2]
    hessian_u_3 = hessian_u[..., 3]
    BH_u = np.sum(np.sqrt(hessian_u_0**2 + hessian_u_1**2 + hessian_u_2**2 + hessian_u_3**2))
    return BH_u

def div2(v):
    div2_v = x_2_mixed_directions(v[..., 0]) + xy_minus(v[..., 1]) + xy_minus(v[..., 2]) + y_2_mixed_directions(v[..., 3])
    return div2_v

# Fourth order operators:

# Case I: y_m y_p x_m y_m = y_m y_p y_m x_m:
#             u_{i-2,j-1} - u_{i-2,j}
#          -3(u_{i-1,j-1} - u_{i-1,j})
#          +3(u_{i,j-1} - u_{i,j})
#           -(u_{i+1,j-1} - u_{i+1,j})
def case_i(u):
    m, n = u.shape
    Du = np.zeros((m, n))
    Du[2:m, 1:n] = u[0:m-2, 0:n-1] - u[0:m-2, 1:n] - 3*(u[1:m-1, 0:n-1] - u[1:m-1, 1:n]) + 3*(u[2:m, 0:n-1] - u[2:m, 1:n]) - (u[3:m+1, 0:n-1] - u[3:m+1, 1:n])
    Du[0, 1:n] = u[m-2, 0:n-1] - u[m-2, 1:n] - 3*(u[m-1, 0:n-1] - u[m-1, 1:n]) + 3*(u[0, 0:n-1] - u[0, 1:n]) - (u[1, 0:n-1] - u[1, 1:n])
    Du[1, 1:n] = u[m-1, 0:n-1] - u[m-1, 1:n] - 3*(u[0, 0:n-1] - u[0, 1:n]) + 3*(u[1, 0:n-1] - u[1, 1:n]) - (u[2, 0:n-1] - u[2, 1:n])
    Du[0, 0] = u[m-2, n-1] - u[m-2, 0] - 3*(u[m-1, n-1] - u[m-1, 0]) + 3*(u[0, n-1] - u[0, 0]) - (u[1, n-1] - u[1, 0])
    Du[1, 0] = u[m-1, n-1]  - u[m-1, 0] - 3*(u[0, n-1]  - u[0, 0]) + 3*(u[1, n-1]  - u[1, 0]) - (u[2, n-1]  - u[2, 0])
    return Du

# Case II: x_p y_p y_p y_m = y_p x_p y_p y_m:
#             u_{i-1,j} - u_{i-1,j+1}
#          -3(u_{i,j} - u_{i,j+1})
#          +3(u_{i+1,j} - u_{i+1,j+1})
#           -(u_{i+2,j} - u_{i+2,j+1})
def case_ii(u):
    m, n = u.shape
    Du = np.zeros((m, n))
    Du[1:m - 2, 0:n - 1] = u[0:m-1, 0:n - 1] - u[0:m-1, 1:n] - 3*(u[1:m - 2, 0:n - 1] - u[1:m - 2, 1:n]) + 3*(u[2:m - 1, 0:n - 1] - u[2:m - 1, 1:n]) - (u[3:m, 0:n - 1] - u[3:m, 1:n])
    Du[0, 0:n - 1] = u[m-1, 0:n-1] - u[m-1, 1:n] - 3*(u[0, 0:n-1] - u[0, 1:n]) + 3*(u[1, 0:n-1] - u[1, 1:n]) - (u[2, 0:n-1] - u[2, 1:n])
    Du[m - 2, 0:n - 1] = u[m-3, 0:n-1] - u[m-3, 1:n] - 3*(u[m-2, 0:n-1] - u[m-2, 1:n]) + 3*(u[m-1, 0:n-1] - u[m-1, 1:n]) - (u[0, 0:n-1] - u[0, 1:n])
    Du[m - 1, 0:n - 1] = u[m - 2, 0:n - 1] - u[m - 2, 1:n] - 3*(u[m - 1, 0:n - 1] - u[m - 1, 1:n]) + 3*(u[0, 0:n - 1] - u[0, 1:n]) - (u[1, 0:n - 1] - u[1, 1:n])
    Du[1:m - 2, n-1] = u[0:m - 3, n-1] - u[0:m - 3, 0] - 3*(u[1:m - 2, n-1] - u[1:m - 2, 0]) + 3*(u[2:m - 1, n-1] - u[2:m - 1, 0]) - (u[3:m, n-1] - u[3:m, 0])
    Du[0, n - 1] = u[m-1, n - 1] - u[m-1, 0] - 3*(u[0, n - 1] - u[0, 0]) + 3*(u[1, n - 1] - u[1, 0]) - (u[2, n - 1] - u[2, 0])
    Du[m - 2, n - 1] = u[m - 3, n - 1] - u[m - 3, 0] - 3*(u[m - 2, n - 1] - u[m - 2, 0]) + 3*(u[m - 1, n - 1] - u[m - 1, 0]) - (u[0, n - 1] - u[0, 0])
    Du[m - 1, n - 1] = u[m - 2, n - 1] - u[m - 2, 0] - 3*(u[m - 1, n - 1] - u[m - 1, 0]) + 3*(u[0, n - 1] - u[0, 0]) - (u[1, n - 1] - u[1, 0])
    return Du

# Case III: x_p y_p x_p x_m = y_p x_p x_p x_m:
#             u_{i,j-1} - u_{i+1,j-1}
#          -3(u_{i,j} - u_{i+1,j})
#          +3(u_{i,j+1} - u_{i+1,j+1})
#           -(u_{i,j+2} - u_{i+1,j+2})
def case_iii(u):
    m, n = u.shape
    Du = np.zeros((m, n))
    Du[0:m-1, 1:n-2] = u[0:m-1, 0:n-3] - u[1:m, 0:n-3] - 3*(u[0:m-1, 1:n-2] - u[1:m, 1:n-2]) + 3*(u[0:m-1, 2:n-1] - u[1:m, 2:n-1]) - (u[0:m-1, 3:n] - u[1:m, 3:n])
    Du[m-1, 1:n-2] = u[m-1, 0:n-3] - u[2:m, 0:n-3] - 3*(u[m-1, 1:n-2] - u[2:m, 1:n-2]) + 3*(u[m-1, 2:n-1] - u[2:m, 2:n-1]) - (u[m-1, 3:n] - u[2:m, 3:n])
    Du[0:m-1, 0] = u[0:m-1, n-1] - u[1:m, n-1] - 3*(u[0:m-1, 0] - u[1:m, 0]) + 3*(u[0:m-1, 1] - u[1:m, 1]) - (u[0:m-1, 2] - u[1:m, 2])
    Du[0:m-1, n-2] = u[0:m-1, n-3] - u[1:m, n-3] - 3*(u[0:m-1, n-2] - u[1:m, n-2]) + 3*(u[0:m-1, n-1] - u[1:m, n-1]) - (u[0:m-1, 0] - u[1:m, 0])
    Du[0:m-1, n-1] = u[0:m-1, n-2] - u[1:m, n-2] - 3*(u[0:m-1, n-1] - u[1:m, n-1]) + 3*(u[0:m-1, 0] - u[1:m, 0]) - (u[0:m-1, 1] - u[1:m, 1])
    Du[m-1, 0] = u[m-1, n-1] - u[0, n-1] - 3*(u[m-1, 0] - u[0, 0]) + 3*(u[m-1, 1] - u[0, 1]) - (u[m-1, 2] - u[0, 2])
    Du[m-1, n-2] = u[m-1, n-3] - u[0, n-3] - 3*(u[m-1, n-2] - u[0, n-2]) + 3*(u[m-1, n-1] - u[0, n-1]) - (u[m-1, 0] - u[0, 0])
    Du[m-1, n-1] = u[m-1, n-2] - u[0, n-2] - 3*(u[m-1, n-1] - u[0, n-1]) + 3*(u[m-1, 0] - u[0, 0]) - (u[m-1, 1] - u[0, 1])
    return Du

# Case IV: x_m x_p x_m y_m = x_m x_p y_m x_m:
#             u_{i-1,j-2} - u_{i,j-2}
#          -3(u_{i-1,j-1} - u_{i,j-1})
#          +3(u_{i-1,j} - u_{i,j})
#           -(u_{i-1,j+1} - u_{i,j+1})
def case_iv(u):
    m, n = u.shape
    Du = np.zeros((m, n))
    Du[1:m, 2:n-1] = u[0:m-1, 0:n-3] - u[1:m, 0:n-3] - 3*(u[0:m-1, 1:n-2] - u[1:m, 1:n-2]) + 3*(u[0:m-1, 2:n-1] - u[1:m, 2:n-1]) - (u[0:m-1, 3:n] - u[1:m, 3:n])
    Du[0, 2:n-1] = u[m-1, 0:n-3] - u[0, 0:n-3] - 3*(u[m-1, 1:n-2] - u[0, 1:n-2]) + 3*(u[m-1, 2:n-1] - u[0, 2:n-1]) - (u[m-1, 3:n] - u[0, 3:n])
    Du[1:m, 0] = u[0:m-1, n-2] - u[1:m, n-2] - 3*(u[0:m-1, n-1] - u[1:m, n-1]) + 3*(u[0:m-1, 0] - u[1:m, 0]) - (u[0:m-1, 1] - u[1:m, 1])
    Du[1:m, 1] = u[0:m-1, n-1] - u[1:m, n-1] - 3*(u[0:m-1, 0] - u[1:m, 0]) + 3*(u[0:m-1, 1] - u[1:m, 1]) - (u[0:m-1, 2] - u[1:m, 2])
    Du[1:m, n-1] = u[0:m-1, n-3] - u[1:m, n-3] - 3*(u[0:m-1, n-2] - u[1:m, n-2]) + 3*(u[0:m-1, n-1] - u[1:m, n-1]) - (u[0:m-1, 0] - u[1:m, 0])
    Du[0, 0] = u[m-1, n-2] - u[0, n-2] - 3*(u[m-1, n-1] - u[0, n-1]) + 3*(u[m-1, 0] - u[0, 0]) - (u[m-1, 1] - u[0, 1])
    Du[0, 1] = u[m-1, n-1] - u[0, n-1] - 3*(u[m-1, 0] - u[0, 0]) + 3*(u[m-1, 1] - u[0, 1]) - (u[m-1, 2] - u[0, 2])
    Du[0, n-1] = u[m-1, n-3] - u[0, n-3] - 3*(u[m-1, n-2] - u[0, n-2]) + 3*(u[m-1, n-1] - u[0, n-1]) - (u[m-1, 0] - u[0, 0])
    return Du

# Case V: x_p x_m x_m x_p = x_m x_p x_m x_p = x_m x_p x_p x_m:
#             6u_{i,j} + u_{i,j+2} + u_{i,j-2}
#                    -4*(u_{i,j+1} + u_{i,j-1})
def case_v(u):
    m, n = u.shape
    Du = np.zeros((m, n))
    Du[0:m, 2:n-2] = 6*u[0:m, 2:n-2] + u[0:m, 4:n] + u[0:m, 0:n-4] - 4*(u[0:m, 3:n-1] + u[0:m, 1:n-3])
    Du[0:m, 0] = 6*u[0:m, 0] + u[0:m, 2] + u[0:m, n-2] - 4*(u[0:m, 1] + u[0:m, n-1])
    Du[0:m, 1] = 6*u[0:m, 1] + u[0:m, 3] + u[0:m, n-1] - 4*(u[0:m, 2] + u[0:m, 0])
    Du[0:m, n-2] = 6*u[0:m, n-2] + u[0:m, 0] + u[0:m, n-4] - 4*(u[0:m, n-1] + u[0:m, n-3])
    Du[0:m, n-1] = 6*u[0:m, n-1] + u[0:m, 1] + u[0:m, n-3] - 4*(u[0:m, 0] + u[0:m, n-2])
    return Du

# Case VI: y_p y_m y_m y_p = y_m y_p y_m y_p = y_m y_p y_p y_m:
#             6u_{i,j} + u_{i+2,j} + u_{i-2,j}
#                    -4*(u_{i+1,j} + u_{i-1,j})
def case_vi(u):
    m, n = u.shape
    Du = np.zeros((m, n))
    Du[2:m-2, 0:n] = 6*u[2:m-2, 0:n] + u[4:m, 0:n] + u[0:m-4, 0:n] - 4*(u[3:m-1, 0:n] + u[1:m-3, 0:n])
    Du[0, 0:n] = 6*u[0, 0:n] + u[2, 0:n] + u[m-2, 0:n] - 4*(u[1, 0:n] + u[m-1, 0:n])
    Du[1, 0:n] = 6*u[1, 0:n] + u[3, 0:n] + u[m-1, 0:n] - 4*(u[2, 0:n] + u[0, 0:n])
    Du[m-2, 0:n] = 6*u[m-2, 0:n] + u[0, 0:n] + u[m-4, 0:n] - 4*(u[m-1, 0:n] + u[m-3, 0:n])
    Du[m-1, 0:n] = 6*u[m-1, 0:n] + u[1, 0:n] + u[m-3, 0:n] - 4*(u[0, 0:n] + u[m-2, 0:n])
    return Du

# Case VII: y_m y_p x_p x_m = x_p y_p x_m y_m = x_p y_p y_m x_m
#         = y_p x_p x_m y_m = y_p x_p y_m x_m = x_m x_p y_p y_m
#         = x_m x_p y_m y_p = y_m y_p x_m x_p = x_m y_m x_p y_p
#         = y_m x_m y_p x_p:
#             u_{i-1,j-1} - 2u_{i-1,j} + u_{i-1,j+1}
#          -2(u_{i,j-1} - 2u_{i,j} + u_{i,j+1})
#           + u_{i+1,j-1} - 2u_{i+1,j} + u_{i+1,j+1}
def case_vii(u):
    m, n = u.shape
    Du = np.zeros((m, n))
    Du[1:m-1, 1:n-1] = u[0:m-2, 0:n-2] - 2*u[0:m-2, 1:n-1] + u[0:m-2, 2:n] -2*(u[1:m-1, 0:n-2] - 2*u[1:m-1, 1:n-1] + u[1:m-1, 2:n]) + u[2:m, 0:n-2] - 2*u[2:m, 1:n-1] + u[2:m, 2:n]
    Du[0, 1:n-1] = u[m-1, 0:n-2] - 2*u[m-1, 1:n-1] + u[m-1, 2:n] -2*(u[0, 0:n-2] - 2*u[0, 1:n-1] + u[0, 2:n]) + u[1, 0:n-2] - 2*u[1, 1:n-1] + u[1, 2:n]
    Du[m-1, 1:n-1] = u[m-2, 0:n-2] - 2*u[m-2, 1:n-1] + u[m-2, 2:n] -2*(u[m-1, 0:n-2] - 2*u[m-1, 1:n-1] + u[m-1, 2:n]) + u[0, 0:n-2] - 2*u[0, 1:n-1] + u[0, 2:n]
    Du[1:m-1, 0] = u[0:m-2, n-1] - 2*u[0:m-2, 0] + u[0:m-2, 1] -2*(u[1:m-1, n-1] - 2*u[1:m-1, 0] + u[1:m-1, 1]) + u[2:m, n-1] - 2*u[2:m, 0] + u[2:m, 1]
    Du[1:m-1, n-1] = u[0:m-2, n-2] - 2*u[0:m-2, n-1] + u[0:m-2, 0] -2*(u[1:m-1, n-2] - 2*u[1:m-1, n-1] + u[1:m-1, 0]) + u[2:m, n-2] - 2*u[2:m, n-1] + u[2:m, 0]
    Du[0, 0] = u[m-1, n-1] - 2*u[m-1, 0] + u[m-1, 1] -2*(u[0, n-1] - 2*u[0, 0] + u[0, 1]) + u[1, n-1] - 2*u[1, 0] + u[1, 1]
    Du[0, n-1] = u[m-1, n-2] - 2*u[m-1, n-1] + u[m-1, 0] -2*(u[0, n-2] - 2*u[0, n-1] + u[0, 0]) + u[1, n-2] - 2*u[1, n-1] + u[1, 0]
    Du[m-1, 0] = u[m-2, n-1] - 2*u[m-2, 0] + u[m-2, 1] -2*(u[m-1, n-1] - 2*u[m-1, 0] + u[m-1, 1]) + u[0, n-1] - 2*u[0, 0] + u[0, 1]
    Du[m-1, n-1] = u[m-2, n-2] - 2*u[m-2, n-1] + u[m-2, 0] -2*(u[m-1, n-2] - 2*u[m-1, n-1] + u[m-1, 0]) + u[0, n-2] - 2*u[0, n-1] + u[0, 0]
    return Du

def fourth_order_derivative(case, u):
    i = {'y_m y_p x_m y_m', 'y_m y_p y_m x_m'}
    ii = {'x_p y_p y_p y_m', 'y_p x_p y_p y_m'}
    iii = {'x_p y_p x_p x_m', 'y_p x_p x_p x_m'}
    iv = {'x_m x_p x_m y_m', 'x_m x_p y_m x_m'}
    v = {'x_p x_m x_m x_p', 'x_m x_p x_m x_p', 'x_m x_p x_p x_m'}
    vi = {'y_p y_m y_m y_p', 'y_m y_p y_m y_p', 'y_m y_p y_p y_m'}
    vii = {'y_m y_p x_p x_m', 'x_p y_p x_m y_m', 'x_p y_p y_m x_m', 'y_p x_p x_m y_m', 'y_p x_p y_m x_m',
           'x_m x_p y_p y_m', 'x_m x_p y_m y_p', 'y_m y_p x_m x_p', 'x_m y_m x_p y_p', 'y_m x_m y_p x_p'}
    if case in vii:
        Du = case_vii(u)
        return Du
    elif case in vi:
        Du = case_vi(u)
        return Du
    elif case in v:
        Du = case_v(u)
        return Du
    elif case in iv:
        Du = case_iv(u)
        return Du
    elif case in iii:
        Du = case_iii(u)
        return Du
    elif case in ii:
        Du = case_ii(u)
        return Du
    elif case in i:
        Du = case_i(u)
        return Du
    else:
        print('Error case not found!!!')

def delta_delta(u):
    delta_delta_u = (fourth_order_derivative('x_m x_p x_m x_p', u) + fourth_order_derivative('x_m x_p y_m y_p', u)
                     + fourth_order_derivative('y_m y_p x_m x_p', u) + fourth_order_derivative('y_m y_p y_m y_p', u))
    return delta_delta_u

def hessian_div2(v):
    m, n, p = v.shape
    hessian_div2_v = np.zeros((m, n, p))
    hessian_div2_v[..., 0] = (fourth_order_derivative('x_m x_p x_p x_m', v[..., 0]) + fourth_order_derivative('x_m x_p y_m x_m', v[..., 1])
                     + fourth_order_derivative('x_m x_p x_m y_m', v[..., 2]) + fourth_order_derivative('x_m x_p y_p y_m', v[..., 3]))
    hessian_div2_v[..., 1] = (fourth_order_derivative('y_p x_p x_p x_m', v[..., 0]) + fourth_order_derivative('y_p x_p y_m x_m', v[..., 1])
                     + fourth_order_derivative('y_p x_p x_m y_m', v[..., 2]) + fourth_order_derivative('y_p x_p y_p y_m', v[..., 3]))
    hessian_div2_v[..., 2] = (fourth_order_derivative('x_p y_p x_p x_m', v[..., 0]) + fourth_order_derivative('x_p y_p y_m x_m', v[..., 1])
                     + fourth_order_derivative('x_p y_p x_m y_m', v[..., 2]) + fourth_order_derivative('x_p y_p y_p y_m', v[..., 3]))
    hessian_div2_v[..., 3] = (fourth_order_derivative('y_m y_p x_p x_m', v[..., 0]) + fourth_order_derivative('y_m y_p y_m x_m', v[..., 1])
                     + fourth_order_derivative('y_m y_p x_m y_m', v[..., 2]) + fourth_order_derivative('y_m y_p y_p y_m', v[..., 3]))
    return hessian_div2_v

