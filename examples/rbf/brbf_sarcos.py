import numpy as np
import numpy.random as npr

from reg.rbf import BayesianFourierRegressor

from sklearn.metrics import mean_squared_error, r2_score

npr.seed(1337)

D_in = 21

N_train = 10000          # max 44484

def load_sarcos_data(D_in, N_train):
    import scipy as sc
    from scipy import io

    # set test data
    if int(N_train / 5) > 4449:  # max 4449
        N_test = 4449
    else:
        N_test = int(N_train / 5)

    # load all available data
    _train_data = sc.io.loadmat('../../datasets/sarcos/sarcos_inv.mat')['sarcos_inv']
    _test_data = sc.io.loadmat('../../datasets/sarcos/sarcos_inv_test.mat')['sarcos_inv_test']

    data = np.vstack((_train_data, _test_data))

    # shuffle data
    np.random.shuffle(data)

    # scale data
    from sklearn.decomposition import PCA
    input_scaler = PCA(n_components=D_in, whiten=True)
    target_scaler = PCA(n_components=1, whiten=True)

    input_scaler.fit(data[:, :D_in])
    target_scaler.fit(data[:, D_in:D_in+1])

    train_data = {'input': input_scaler.transform(_train_data[:N_train, :D_in]),
                  'target': target_scaler.transform(_train_data[:N_train, D_in:D_in+1])}

    test_data = {'input': input_scaler.transform(_test_data[:N_test, :D_in]),
                 'target': target_scaler.transform(_test_data[:N_test, D_in:D_in+1])}

    train_input = train_data['input']
    train_target = train_data['target']

    test_input = test_data['input']
    test_target = test_data['target']

    return train_input, train_target, test_input, test_target

fn = BayesianFourierRegressor(sizes=[D_in, 200, 1],
                              bandwidth=np.ones((D_in,)))

X_train, Y_train, X_test, Y_test = load_sarcos_data(D_in, N_train)
Y_train, Y_test = np.reshape(Y_train, (N_train, 1)), np.reshape(Y_test, -1)

fn.fit(Y_train, X_train, preprocess=True)

y_pred = np.zeros((len(X_test),))
for n in range(len(X_test)):
    y_pred[n] = fn.predict(X_test[n, :])

mse = mean_squared_error(Y_test, y_pred)
smse = 1. - r2_score(Y_test, y_pred, multioutput='variance_weighted')
print('mse, smse:', mse, smse)

from matplotlib import pyplot as plt
plt.figure()
plt.scatter(Y_test, y_pred)
plt.show()