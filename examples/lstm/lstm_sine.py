import torch
import numpy as np

from reg.nn import LSTMRegressor

to_torch = lambda arr: torch.from_numpy(arr).double()


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    T, L, N = 20, 250, 10

    input_size = 1
    output_size = 1

    x = np.empty((N, L), 'int64')
    x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
    data = np.sin(x / 1.0 / T).astype('float64')

    input = to_torch(data[3:, :-1]).view(7, -1, input_size)
    target = to_torch(data[3:, 1:]).view(7, -1, output_size)

    test_input = to_torch(data[:3, :-1]).view(3, -1, input_size)
    test_target = to_torch(data[:3, 1:]).view(3, -1, output_size)

    lstm = LSTMRegressor(input_size,
                         output_size,
                         nb_neurons=[25, 25])

    lstm.fit(target, input, 50)

    with torch.no_grad():
        future = 1000
        pred = lstm(test_input, future=future)
        y = pred.detach().numpy()

    plt.figure(figsize=(30, 10))
    plt.title('Predict future values for time sequences', fontsize=30)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    def draw(yi, color):
        plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth=2.0)
        plt.plot(np.arange(input.size(1), input.size(1) + future),
                 yi[input.size(1):], color + ':', linewidth=2.0)

    draw(y[0], 'r')
    draw(y[1], 'g')
    draw(y[2], 'b')
    plt.show()
    plt.close()
