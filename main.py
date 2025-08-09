import loader as LD
import parameter_learning_algorithm as PL
import evaluation as EV
from parameter_learning_util_func import regularized_u_plus

regularizer_type = 'TV'
train_data, A, v_set, Iterations, Residuals, Lambda, p, n, N = LD.train_input('initiate', regularizer_type)

PL.HPGCG(train_data, regularizer_type, 25000, 1e-4, 1,A, v_set, Iterations, Residuals, Lambda, p, n, N)

#test_data, TV_A, TL_A, BH_A = LD.evaluation_input()

#EV.evaluation(test_data, TL_A, TL_A, BH_A)