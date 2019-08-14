import autograd.numpy as np

from reg.nn.nn_ag import NNRegressor as agNetwork
from reg.nn.nn_npy import NNRegressor as npNetwork


if __name__ == '__main__':

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import OneHotEncoder

    set = load_breast_cancer()

    x, y = set['data'], set['target']

    enc = OneHotEncoder(categories='auto')
    y = enc.fit_transform(y[:, np.newaxis]).toarray()

    xt, xv, yt, yv = train_test_split(x, y, test_size=0.2)

    scaler = StandardScaler()
    xt = scaler.fit_transform(xt)
    xv = scaler.fit_transform(xv)

    nb_in = x.shape[-1]
    nb_out = y.shape[-1]

    # fit neural network with numpy
    nn_np = npNetwork([nb_in, 4, nb_out], nonlin='tanh', output='logistic', loss='ce')

    print('using numpy network:')
    nn_np.fit(yt, xt, nb_epochs=250, batch_size=64, lr=1e-2)

    _yt = nn_np.forward(xt)
    class_error = np.linalg.norm(yt - np.rint(_yt), axis=1)
    print('numpy', 'train:', 'cost=', nn_np.cost(yt, xt), 'class. error=', np.mean(class_error))

    _yv = nn_np.forward(xv)
    class_error = np.linalg.norm(yv - np.rint(_yv), axis=1)
    print('numpy', 'test:', 'cost=', nn_np.cost(yv, xv), 'class. error=', np.mean(class_error))

    # fit neural network with autograd
    nn_ag = agNetwork([nb_in, 4, nb_out], nonlin='tanh', output='logistic', loss='ce')

    print('using autograd network:')
    nn_ag.fit(yt, xt, nb_epochs=250, batch_size=64, lr=1e-2)

    _yt = nn_ag.forward(xt)
    class_error = np.linalg.norm(yt - np.rint(_yt), axis=1)
    print('autograd', 'train:', 'cost=', nn_ag.cost(yt, xt), 'class. error=', np.mean(class_error))

    _yv = nn_ag.forward(xv)
    class_error = np.linalg.norm(yv - np.rint(_yv), axis=1)
    print('autograd', 'test:', 'cost=', nn_ag.cost(yv, xv), 'class. error=', np.mean(class_error))
