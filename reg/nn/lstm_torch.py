import torch
import torch.nn as nn
from torch.optim import LBFGS
import numpy as np

to_torch = lambda arr: torch.from_numpy(arr).float()


class LSTMRegressor(nn.Module):
    def __init__(self, input_size, output_size, nb_neurons):
        super(LSTMRegressor, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.nb_neurons = nb_neurons

        self.lstm0 = nn.LSTMCell(self.input_size, self.nb_neurons[0])
        self.lstm1 = nn.LSTMCell(self.nb_neurons[0], self.nb_neurons[1])

        self.linear = nn.Linear(self.nb_neurons[1], self.output_size)

        self.criterion = nn.MSELoss()
        self.optim = None

    def forward(self, input, future=0):
        outputs = []
        ht = torch.zeros(input.size(0), self.nb_neurons[0], dtype=torch.double)
        ct = torch.zeros(input.size(0), self.nb_neurons[0], dtype=torch.double)

        gt = torch.zeros(input.size(0), self.nb_neurons[1], dtype=torch.double)
        bt = torch.zeros(input.size(0), self.nb_neurons[1], dtype=torch.double)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            ht, ct = self.lstm0(input_t, (ht, ct))
            gt, bt = self.lstm1(ht, (gt, bt))
            output = self.linear(gt)
            outputs += [output]

        for i in range(future):
            ht, ct = self.lstm0(output, (ht, ct))
            gt, bt = self.lstm1(ht, (gt, bt))
            output = self.linear(gt)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)

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
