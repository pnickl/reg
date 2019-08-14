import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from scipy import spatial

np.random.seed(1337)

sigma = 0.3
x = np.random.uniform(-10.0, 10.0, size=50)
x_test = np.linspace(-9.0, 9.0, 1000)

y = 0.1 * x**3 + 0.05 * x**2 + np.random.rand(x.shape[0]) * sigma
y_test = 0.1 * x_test**3 + 0.05 * x_test**2

# y = 5 * (0.5 * x**2 + 2 * x + 0.3) + np.random.rand(x.shape[0]) * sigma
# y_test = 5 * (0.5 * x_test**2 + 2 * x_test) + 0.3

n_features = 10
n_states = 1

freq = np.random.randn(n_features,)
phase = np.random.uniform(-np.pi , np.pi, size=n_features)
band = np.median(sc.spatial.distance.pdist(x[:, None]))

phi = np.zeros((x.shape[0], n_features))
phi_test = np.zeros((x_test.shape[0], n_features))

for i in range(n_features):
	phi[:, i] = np.sin(freq[i] * x / band + phase[i])
	phi_test[:, i] = np.sin(freq[i] * x_test / band + phase[i])


plt.figure()
plt.plot(x_test, phi_test)
plt.show()

reg = 1e-4

weights = np.linalg.inv(phi.T @ phi + reg * np.eye(n_features)) @ phi.T @ y
y_val = phi_test @ weights

plt.figure()
plt.scatter(x, y, color='g')
plt.plot(x_test, y_test)
plt.plot(x_test, y_val)
plt.show()

print(np.linalg.norm(y_val - y_test) / y_test.shape[0])