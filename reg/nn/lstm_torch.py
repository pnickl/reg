import torch
import torch.nn as nn

from torch.optim import LBFGS

to_torch = lambda arr: torch.from_numpy(arr).float()


class LSTMRegressor(nn.Module):
    def __init__(self, input_size, output_size, nb_neurons):
        super(LSTMRegressor, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.nb_neurons = nb_neurons
        self.nb_layers = 2

        self.lstm0 = nn.LSTMCell(self.input_size, self.nb_neurons[0])
        self.lstm1 = nn.LSTMCell(self.nb_neurons[0], self.nb_neurons[1])

        self.linear = nn.Linear(self.nb_neurons[1], self.output_size)

        self.criterion = nn.MSELoss()
        self.optim = None

    def forward(self, input, future=0):
        outputs = []

        batch_size = input.size(0)
        nb_samples = input.size(1)

        ht = torch.zeros(batch_size, self.nb_neurons[0], dtype=torch.double)
        ct = torch.zeros(batch_size, self.nb_neurons[0], dtype=torch.double)

        gt = torch.zeros(batch_size, self.nb_neurons[1], dtype=torch.double)
        bt = torch.zeros(batch_size, self.nb_neurons[1], dtype=torch.double)

        for i in range(nb_samples):
            ht, ct = self.lstm0(input[:, i, :], (ht, ct))
            gt, bt = self.lstm1(ht, (gt, bt))
            output = self.linear(gt)
            outputs += [output]

        for i in range(future):
            ht, ct = self.lstm0(output, (ht, ct))
            gt, bt = self.lstm1(ht, (gt, bt))
            output = self.linear(gt)
            outputs += [output]

        outputs = torch.stack(outputs, 1)

        return outputs

    def fit(self, y, x, nb_epochs, lr=0.5):
        self.double()

        self.optim = LBFGS(self.parameters(), lr=lr)
        for n in range(nb_epochs):

            def closure():
                self.optim.zero_grad()
                _y = self(x)
                loss = self.criterion(_y, y)
                loss.backward()

                print('Epoch: {}/{}.............'.format(n, nb_epochs), end=' ')
                print("Loss: {:.6f}".format(loss.item()))

                return loss

            self.optim.step(closure)
