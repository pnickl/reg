import numpy as np
import numpy.random as npr

from reg.rbf import BayesianFourierRegressor

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

npr.seed(1337)

# prepare data
set = load_boston()
x, y = set['data'], set['target']
y = y[:, np.newaxis]

x_train, x_test, y_train, y_test \
    = train_test_split(x, y, test_size=0.1)

fn = BayesianFourierRegressor(sizes=[13, 512, 1],
                              bandwidth=np.ones((13,)))

fn.fit(y_train, x_train, preprocess=True)

y_pred = np.zeros((len(x_test),))
for n in range(len(x_test)):
    y_pred[n] = fn.predict(x_test[n, :])

mse = mean_squared_error(y_test, y_pred)
smse = 1. - r2_score(y_test, y_pred, multioutput='variance_weighted')
print(mse, smse)
