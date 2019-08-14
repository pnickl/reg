import math
import torch
import gpytorch

from reg.gp import GPRegressor


if __name__ == "__main__":

    # data
    input = torch.linspace(0, 1, 100)
    output = torch.sin(input * (2 * math.pi)) + torch.randn(input.size()) * 0.1

    # build model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPRegressor(input, output, likelihood)

    # fit gp
    model.fit()

    # k-step mean squared error
    mse, norm_mse = model.kstep_mse([input], [output])
    print(mse, norm_mse)
