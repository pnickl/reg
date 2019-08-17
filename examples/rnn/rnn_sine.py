import torch
from torch import nn
import autograd.numpy as np

from reg.nn import RNNRegressor

to_torch = lambda arr: torch.from_numpy(arr).float()


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    T, L, N = 20, 250, 10

    input_size = 1
    output_size = 1

    x = np.empty((N, L), 'int64')
    x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
    data = np.sin(x / 1.0 / T).astype('float64')

    input = to_torch(data[:, :-1]).view(N, -1, input_size)
    target = to_torch(data[:, 1:]).view(N, -1, output_size)

    nb_epcohs = 1000
    nb_layers = 1
    hidden_size = 25
    nb_samples = input.shape[0]

    rnn = RNNRegressor(input_size=input_size,
                       output_size=output_size,
                       hidden_size=hidden_size,
                       nb_layers=nb_layers,
                       criterion=nn.MSELoss())

    rnn.fit(target, input, nb_epochs=3500, lr=5.e-4)

    predict = []
    for t in range(input.shape[1]):
        aux, _ = rnn(input[:, t].view(-1, 1, input_size))
        predict.append(aux)

    _target = np.array(target)
    _predict = np.array(np.concatenate([_x.detach().numpy() for _x in predict], axis=1))

    plt.plot(_target[0, ...], label='target')
    plt.plot(_predict[0, ...], label='prediction')
    plt.legend()
    plt.show()
