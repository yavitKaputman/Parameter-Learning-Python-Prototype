import numpy as np
import scipy as sp

input_directory = 'data/input/'

def train_input(command, regularizer_type):
    train_data = sp.io.loadmat(input_directory + 'train_data.mat')['train_data']
    _, p, _, N = train_data.shape
    if command == 'initiate' and regularizer_type == 'TV':
        A = np.zeros((257,257))
        v_set = np.zeros((p, p, 2, N))
        Iterations = [[0]]
        Residuals = [[1]]

    elif command == 'initiate' and regularizer_type == 'TL':
        A = np.zeros((257, 257))
        A[0, 0] = 1
        v_set = np.zeros((p, p, N))
        Iterations = [[0]]
        Residuals = [[1]]

    elif command == 'initiate' and regularizer_type == 'BH':
        A = np.zeros((257, 257))
        A[0, 0] = 1
        v_set = np.zeros((p, p, 4, N))
        Iterations = [[0]]
        Residuals = [[1]]

    elif command == 'continue':
        train_data = sp.loadmat(f'data/output/trained data/{regularizer_type}_trained_data.mat')
        A = train_data[f'{regularizer_type}_A']
        v_set = train_data[f'{regularizer_type}_v_set']
        Iterations = train_data[f'{regularizer_type}_Iterations']
        Residuals = train_data[f'{regularizer_type}_Residuals']
    else:
        print("Error, regularizer type must be TV or TL or BH and command must be initiate or continue!!!")

    train_data = np.asarray(train_data)
    A = np.asarray(A)
    v_set = np.asarray(v_set)
    Iterations = np.asarray(Iterations)
    Residuals = np.asarray(Residuals)

    Lambda = 50

    n = p**2

    return train_data, A, v_set, Iterations, Residuals, Lambda, p, n, N


def evaluation_input():
    test_data = sp.io.loadmat(input_directory + 'test_data.mat')['test_data']
    test_data = np.copy(test_data.squeeze())
    TV_trained_data = sp.loadmat('data/output/trained data/TV_trained_data.mat')
    TL_trained_data = sp.loadmat('data/output/trained data/TL_trained_data.mat')
    BH_trained_data = sp.loadmat('data/output/trained data/BH_trained_data.mat')
    TV_A = TV_trained_data['TV_A']
    TL_A = TL_trained_data['TL_A']
    BH_A = BH_trained_data['BH_A']
    return test_data, TV_A, TL_A, BH_A



