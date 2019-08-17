from reg.nn.pn_npy import Perceptron


if __name__ == '__main__':

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    from sklearn.preprocessing import StandardScaler

    set = load_breast_cancer()
    x, y = set['data'], set['target']

    xt, xv, yt, yv = train_test_split(x, y, test_size=0.2)

    scaler = StandardScaler()
    xt = scaler.fit_transform(xt)
    xv = scaler.fit_transform(xv)

    nb_in = x.shape[-1]

    perc = Perceptron(nb_in)
    perc.fit(yt, xt, nb_epochs=250, lr=0.25)

    print("testing", "class error=", perc.error(yv, xv))
