import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from rl.hyreps.v1 import MultiLogistic
from scipy.special import logit

from sklearn.linear_model import Ridge

from misc.beautify import beautify

import seaborn as sns

from matplotlib.colors import ListedColormap

red_cmap = ListedColormap(sns.color_palette("OrRd", 10).as_hex())
green_cmap = ListedColormap(sns.color_palette("BuGn", 10).as_hex())
blue_cmap = ListedColormap(sns.color_palette("PuBu", 10).as_hex())

class LogitRegression:

	def __init__(self, x, y, **kwargs):

		self.n_states = x.shape[-1]
		self.n_regions = y.shape[-1]

		self.x = x
		self.y = y

		if 'band' in kwargs:
			self.band = kwargs.get('band', False)
			self.n_feat = kwargs.get('n_feat', False)
			self.lgstc_mdl = MultiLogistic(self.n_states, self.n_regions,
										   band=self.band, n_feat=self.n_feat)
		else:
			self.degree = kwargs.get('degree', False)
			self.lgstc_mdl = MultiLogistic(self.n_states, self.n_regions,
										   degree=self.degree)

	def regress(self):
		reg = Ridge(alpha=1e-8, fit_intercept=False,
					tol=1e-6, solver="auto", max_iter=None)

		feat = self.lgstc_mdl.features(self.x)

		for n in range(self.n_regions):
			y = self.y[:, n, :]
			lgt = logit(np.clip(y, 0.0001, 0.9999))
			reg.fit(feat, lgt)

			self.lgstc_mdl.par[n, :] = np.reshape(reg.coef_, (self.lgstc_mdl.n_feat * self.n_regions))


	def plot(self, x_min=-5.0, x_max=5.0, npts=2500):
		maps = [red_cmap, green_cmap, blue_cmap]

		x1 = np.linspace(x_min, x_max, npts)
		x2 = np.linspace(x_min, x_max, npts)
		X1, X2 = np.meshgrid(x1, x2)

		X = np.dstack((X1, X2))
		Z = self.lgstc_mdl.transitions(X)

		for n in range(self.n_regions):
			ax = plt.figure().gca()
			for m in range(self.n_regions):
				masked = np.ma.array(Z[:, :, n, m], mask=(np.argmax(Z[:, :, n, :], axis=-1) != m))
				ax.imshow(masked, extent=[x_min, x_max, x_min, x_max],
						  alpha=1.0, aspect='equal',
						  # cmap=maps[m], norm=colors.LogNorm(vmin=0.35, vmax=masked.max()))
							cmap = maps[m], norm = colors.LogNorm(vmin=0.3, vmax=1.0))

			ax = beautify(ax)
			ax.grid(False)

			ax.set_xlim([-x_max, x_max])
			ax.set_ylim([-x_max, x_max])

			plt.autoscale()
			plt.show()


if __name__ == "__main__":

	# generate data
	x_min, x_max = -5.0, 5.0
	n_states, n_regions  = 2, 2
	n_points = 500

	mu = np.array([0.0, 0.0])
	sigma = np.array([[1.0, 0.1], [0.1, 1.0]])
	x = np.random.multivariate_normal(mu, sigma, size=n_points)

	lgstc_mdl = MultiLogistic(n_states, n_regions, degree=1)
	y = lgstc_mdl.transitions(x)

	# fit and plot
	lgt_reg = LogitRegression(x, y, degree=1)
	# lgt_reg.regress()

	for n in range(n_regions):
		lgt_reg.lgstc_mdl.par[n, ...] = np.array([1.0, 0.1, 0.1, 1.0, -35.0, 35.0])

	lgt_reg.plot()

	from matplotlib2tikz import save as tikz_save
	tikz_save("hybrid_lgstc.tex")
