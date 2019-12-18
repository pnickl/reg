import numpy as np

import scipy as sc
from scipy import special

import numexpr as ne

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap

from matplotlib import rc


import seaborn as sns

EXP_MAX = 700.0
EXP_MIN = -700.0

red_cmap = ListedColormap(sns.color_palette("OrRd", 3, 1).as_hex())
green_cmap = ListedColormap(sns.color_palette("BuGn", 3, 1).as_hex())
blue_cmap = ListedColormap(sns.color_palette("PuBu", 3, 1).as_hex())

rc('lines', **{'linewidth': 1})
rc('text', usetex=True)


def beautify(ax):
    ax.set_frame_on(True)
    ax.minorticks_on()

    ax.grid(True)
    ax.grid(linestyle=':')

    ax.tick_params(which='both', direction='in',
                   bottom=True, labelbottom=True,
                   top=True, labeltop=False,
                   right=True, labelright=False,
                   left=True, labelleft=True)

    ax.tick_params(which='major', length=6)
    ax.tick_params(which='minor', length=3)

    ax.autoscale(tight=True)
    # ax.set_aspect('equal')

    if ax.get_legend():
        ax.legend(loc='best')

    return ax


class FourierFeatures:

    def __init__(self, nb_states, nb_feat, band):
        self.nb_states = nb_states
        self.nb_feat = nb_feat

        self.freq = np.random.multivariate_normal(mean=np.zeros(self.nb_states),
												  cov=np.diag(1.0 / band), size=self.nb_feat)

        self.shift = np.random.uniform(-np.pi, np.pi, size=self.nb_feat)

    def fit_transform(self, x):
        phi = np.sin(np.einsum('nk,...k->...n', self.freq, x) + self.shift)
        phi = np.insert(phi, 0, 1.0, axis=-1)
        return phi


class MultiLogistic:

    def __init__(self, nb_states, nb_regions, **kwargs):
        self.nb_states = nb_states
        self.nb_regions = nb_regions

        if 'band' in kwargs:
            self.band = kwargs.get('band', False)
            self.nb_feat = kwargs.get('n_feat', False)
            self.basis = FourierFeatures(self.nb_states, self.nb_feat, self.band)
        else:
            self.degree = kwargs.get('degree', False)
            self.nb_feat = int(sc.special.comb(self.degree + self.nb_states, self.degree))
            self.basis = PolynomialFeatures(self.degree)

        self.par = np.random.randn(self.nb_regions, self.nb_regions * self.nb_feat)

    def features(self, x):
        feat = self.basis.fit_transform(x.reshape(-1, self.nb_states))
        return np.reshape(feat, x.shape[:-1] + (self.nb_feat,))

    def transitions(self, x, par=None):
        if par is None:
            par = self.par

        feat = self.features(x)
        trans = self.logistic(par.reshape((self.nb_regions, self.nb_regions, -1)), feat)

        return trans

    def logistic(self, p, feat):
        a = np.einsum('...k,mlk->...ml', feat, p)
        a = np.clip(a, EXP_MIN, EXP_MAX)
        expa = ne.evaluate('exp(a)')
        l = expa / np.sum(expa, axis=-1, keepdims=True)
        return np.squeeze(l)


class LogitRegression:

    def __init__(self, x, y, **kwargs):

        self.n_states = x.shape[-1]
        self.n_regions = y.shape[-1]

        self.x = x
        self.y = y

        if 'band' in kwargs:
            self.band = kwargs.get('band', False)
            self.n_feat = kwargs.get('n_feat', False)
            self.lgstc_mdl = MultiLogistic(self.n_states, self.n_regions, band=self.band, n_feat=self.n_feat)
        else:
            self.degree = kwargs.get('degree', False)
            self.lgstc_mdl = MultiLogistic(self.n_states, self.n_regions, degree=self.degree)

    def regress(self):
        reg = Ridge(alpha=1e-8, fit_intercept=False, tol=1e-6, solver="auto", max_iter=None)

        feat = self.lgstc_mdl.features(self.x)

        for n in range(self.n_regions):
            y = self.y[:, n, :]
            lgt = sc.special.logit(np.clip(y, 0.0001, 0.9999))
            reg.fit(feat, lgt)

            self.lgstc_mdl.par[n, :] = np.reshape(reg.coef_, (self.lgstc_mdl.nb_feat * self.n_regions))

    def plot(self, x_min=-10.0, x_max=10.0, npts=2500):
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
                ax.imshow(masked, extent=[x_min, x_max, x_min, x_max], alpha=1., aspect='equal',
						  # cmap=maps[m], norm=colors.LogNorm(vmin=0.35, vmax=masked.max()))
                          cmap=maps[m], norm=colors.Normalize(vmin=0.3, vmax=1.))

            ax = beautify(ax)
            ax.grid(False)

            ax.set_xlim([x_min, x_max])
            ax.set_ylim([x_min, x_max])

            plt.autoscale()
            plt.show()


if __name__ == "__main__":

    # generate data
    x_min, x_max = -10.0, 10.0
    nb_states, nb_regions = 2, 3
    nb_points = 1000

    mu = np.array([0.0, 0.0])
    sigma = np.array([[1.0, 0.1], [0.1, 1.0]])
    x = np.random.multivariate_normal(mu, sigma, size=nb_points)

    lgstc_mdl = MultiLogistic(nb_states, nb_regions, degree=3)
    y = lgstc_mdl.transitions(x)

    # fit and plot
    lgt_reg = LogitRegression(x, y, degree=3)
    lgt_reg.regress()

    lgt_reg.plot()

    # from tikzplotlib import save as tikz_save
    # tikz_save("hybrid_lgstc_cube.tex")
