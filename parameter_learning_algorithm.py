import numpy as np
import parameter_learning_util_func as PLUF

def HPGCG(train_data, regularizer_type, max_iter, least_res, backup_iter,A, v_set, Iterations, Residuals, Lambda, p, n, N):
    u_plus_set = train_data[..., 0, :]
    g_set = train_data[..., 1, :]
    g_bar_set = PLUF.g_bar(g_set, n, N)
    regularized_u_plus_set = PLUF.regularized_u_plus(regularizer_type, u_plus_set, N)
    frequent_argument_set, c_set = PLUF.iterative_calculations(regularizer_type, regularized_u_plus_set, g_set, v_set, p, N)
    res = Residuals[0, -1]
    iteration = Iterations[0, -1]
    while res >= least_res and iteration <= max_iter:
        iteration += 1
        Iterations = np.append(Iterations, iteration)
        A_tilde = PLUF.A_Tilde(A, c_set, g_bar_set, Lambda, N)
        print(A_tilde)
        v_tilde_set = PLUF.v_tilde(regularizer_type, frequent_argument_set, g_bar_set, A_tilde, p, N)
        A_direction = A - A_tilde
        v_direction_set = v_set - v_tilde_set
        res = PLUF.Res(regularizer_type, regularized_u_plus_set, frequent_argument_set, g_bar_set, A_direction, v_direction_set, Lambda, N)
        Residuals = np.append(Residuals, res)
        theta = PLUF.Theta(regularizer_type, A_direction, v_direction_set, N, res, Lambda)
        A -= theta * A_direction
        v_set -= theta * v_direction_set
        frequent_argument_set, c_set = PLUF.iterative_calculations(regularizer_type, regularized_u_plus_set, g_set, v_set, p, N)
        if iteration == 1:
            print(f"{regularizer_type} Parameter Learning Initial Iteration {iteration}, Initial Residual: {res:.6f}")
        elif iteration % backup_iter == 0:
            print(f"{regularizer_type} Parameter Learning Iteration {iteration}, Residual: {res:.6f}")
            PLUF.save_trained_data(regularizer_type, A, v_set, Iterations, Residuals)
    print(f"{regularizer_type} Parameter Learning Final Iteration {iteration}, Final Residual: {res:.6f}")
    PLUF.save_trained_data(regularizer_type, A, v_set, Iterations, Residuals)