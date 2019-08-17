import torch
from torch import nn
from torch.optim import Adam


class RNNRegressor(nn.Module):
    def __init__(self, input_size, output_size,
                 hidden_size, nb_layers,
                 nonlinearity='tanh'):
        super(RNNRegressor, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.nb_layers = nb_layers

        self.nonlinearity = nonlinearity

        self.rnn = nn.RNN(input_size, hidden_size,
                          nb_layers, batch_first=True,
                          nonlinearity=nonlinearity)

        self.linear = nn.Linear(hidden_size, output_size)

        self.criterion = nn.MSELoss()
        self.optim = None

    def forward(self, x):
        batch_size = x.size(0)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.rnn(x, hidden)

        out = out.contiguous().view(batch_size, -1, self.hidden_size)
        out = self.linear(out)

        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.nb_layers, batch_size, self.hidden_size)

    def fit(self, y, x, nb_epochs, lr=1.e-3):
        self.optim = Adam(self.parameters(), lr=lr)

        for n in range(nb_epochs):
            self.optim.zero_grad()
            _y, hidden = self(x)
            loss = self.criterion(_y.view(-1, self.output_size),
                                  y.view(-1, self.output_size))
            loss.backward()
            self.optim.step()

            if n % 10 == 0:
                print('Epoch: {}/{}.............'.format(n, nb_epochs), end=' ')
                print("Loss: {:.6f}".format(loss.item()))
