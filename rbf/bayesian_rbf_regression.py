import numpy as np
import scipy as sc
import matplotlib.pyplot as plt


np.random.seed(1337)

sigma = 0.3
x = np.random.rand(20,) * 10 - 8
y = 0.1 * x**3 + 0.05 * x**2 + np.random.rand(x.shape[0]) * sigma

x_test = np.linspace(-9.0, 9.0, 1000)
y_test = 0.1 * x_test**3 + 0.05 * x_test**2

n_centers = 10
centers = np.linspace(-7.0, 7.0, n_centers)
bandwidth = 7.0

phi = np.zeros((x.shape[0], n_centers))
phi_test = np.zeros((x_test.shape[0], n_centers))

for i in range(n_centers):
	phi[:, i] = np.exp(-0.5 * (x - centers[i])**2 / bandwidth**2)
	phi_test[:, i] = np.exp(-0.5 * (x_test - centers[i]) ** 2 / bandwidth**2)

plt.figure()
plt.plot(x_test, phi_test)
plt.show()

reg = 1e-4

mu_weights = np.linalg.inv(phi.T @ phi + reg * np.eye(n_centers) * sigma**2) @ phi.T @ y
sigma_weights = sigma**2 * np.linalg.inv(phi.T @ phi + reg * np.eye(n_centers) * sigma**2)

mu_y_val = phi_test @ mu_weights
sigma2_y_val = np.zeros(mu_y_val.shape[0])

plt.figure()
plt.plot(x_test, y_test)
plt.plot(x_test, mu_y_val)
plt.show()

for i in range(mu_y_val.shape[0]):
	sigma2_y_val[i] = sigma**2 * (1.0 + phi_test[i, :] @ sigma_weights @ phi_test[i, :].T)

