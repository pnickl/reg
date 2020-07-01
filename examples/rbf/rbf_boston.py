import numpy as np
import numpy.random as npr

from reg.rbf import FourierRegressor

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

npr.seed(1337)

# prepare data
set = load_boston()
x, y = set['data'], set['target']
y = y[:, np.newaxis]

x_train, x_test, y_train, y_test \
    = train_test_split(x, y, test_size=0.1)

fn = FourierRegressor(sizes=[13, 512, 1], bandwidth=np.ones((13, )))

fn.fit(y_train, x_train, nb_epochs=5000, preprocess=True)
y_pred = fn.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
print(mse)
