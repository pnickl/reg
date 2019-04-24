import GPy
import numpy as np

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

hr = 50
rs = 100
states = 4
actions = 1

Xt = np.load('data/pendulum/pendulum_state_train.npy')
Ut = np.load('data/pendulum/pendulum_action_train.npy')

Xin = np.reshape(Xt[:rs, 0:hr-1, :], (-1, states), order='C')
Uin = np.reshape(Ut[:rs, 0:hr-1, :], (-1, actions), order='C')

input = np.concatenate((Xin, Uin), axis=1)
output = np.reshape(Xt[:rs, 1:hr, :], (-1, states), order='C')

# create simple GP Model
Z = np.random.rand(100, states + actions)
m = GPy.models.SparseGPRegression(input, output, Z=Z)
# m = GPy.models.GPRegression(input, output)

m.optimize('bfgs', max_iters=200)

# np.save("../data/pendulum/pendulum_gpy.npy", m.param_array)

print("Finished Training")

# test on training data
xt = Xt[:rs, :hr, :]
ut = Ut[:rs, :hr, :]

xp = np.zeros((rs, hr, states))
xp[:, 0, :] = xt[:rs, 0, :]

for n in range(rs):
	for t in range(1, hr):
		input = np.concatenate((xp[n, t - 1, :], ut[n, t - 1, :]), axis=0)
		xp[n, t, :] = m.predict(input[None, :])[0]

error = mean_squared_error(np.reshape(xt, (-1, states)), np.reshape(xp, (-1, states)))
print("Error on training data: ", error)

# # test on test data
Xv = np.load('data/pendulum/pendulum_state_test.npy')
Uv = np.load('data/pendulum/pendulum_action_test.npy')

xv = Xv[:rs, :hr, :]
uv = Uv[:rs, :hr, :]

xp = np.zeros((rs, hr, states))
xp[:, 0, :] = xv[:rs, 0, :]

for n in range(rs):
	for t in range(1, hr):
		input = np.concatenate((xp[n, t - 1, :], uv[n, t - 1, :]), axis=0)
		xp[n, t, :] = m.predict(input[None, :])[0]

error = mean_squared_error(np.reshape(xv, (-1, states)), np.reshape(xp, (-1, states)))
print("Error on testing data: ", error)

rollout = np.random.randint(0, xv.shape[0])
plt.plot(xv[rollout, :, :])
plt.plot(xp[rollout, :, :])
plt.show()
