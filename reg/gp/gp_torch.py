import torch
import gpytorch
import numpy as np


class GPRegressor(gpytorch.models.ExactGP):

	def __init__(self, input, target, likelihood):
		super(GPRegressor, self).__init__(input, target, likelihood)

		self.input = input
		self.target = target

		self.mean_module = gpytorch.means.ConstantMean()
		self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

	def forward(self, x):
		mean_x = self.mean_module(x)
		covar_x = self.covar_module(x)
		return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

	def fit(self, nb_iter=100):
		# Find optimal model hyperparameters
		self.train()
		self.likelihood.train()

		# Use the adam optimizer
		optimizer = torch.optim.Adam([{'params': self.parameters()}], lr=0.1)

		# "Loss" for GPs - the marginal log likelihood
		mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

		for i in range(nb_iter):
			# Zero gradients from previous iteration
			optimizer.zero_grad()
			# Output from model
			_model_output = self(self.input)
			# Calc loss and backprop gradients
			loss = - mll(_model_output, self.target)
			loss.backward()
			print('Iter %d/%d - Loss: %.3f' % (i + 1, nb_iter, loss.item()))
			optimizer.step()

	def kstep_mse(self, input, target, horizon):
		from sklearn.metrics import mean_squared_error, r2_score

		self.eval()
		self.likelihood.eval()

		mse, norm_mse = [], []

		for _input, _target in zip(input, target):
			_nb_steps = _input.shape[0]

			with torch.no_grad(), gpytorch.settings.fast_pred_var():
				_predictive_posterior = self.likelihood(self(_input))

			with torch.no_grad():
				_prediction = _predictive_posterior.mean

			# sum of mean squared errors
			_mse = mean_squared_error(_target.numpy()[horizon - 1:, :],
									  _prediction.numpy()[:_nb_steps - horizon + 1, :])
			mse.append(_mse)

			# normalized sum of mean squared errors
			_norm_mse = r2_score(_target.numpy()[horizon - 1:, :],
								 _prediction.numpy()[:_nb_steps - horizon + 1, :],
								 multioutput='variance_weighted')
			norm_mse.append(_norm_mse)

		return np.mean(mse), np.mean(norm_mse)
