import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from neuralnet import Net


def run():
    df = pd.read_csv("winequality-white.csv", sep=';', decimal=",", dtype=np.float)
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1:].values
    y = OneHotEncoder().fit_transform(y).toarray().astype(float)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    input_layer_size = np.size(x_train, 1)
    output_layer_size = np.size(y_train, 1)
    hidden_layer_size = int((input_layer_size + output_layer_size) / 2)
    topology = [3, 2, 1]
    my_net = Net(topology)

    my_net.feed_forward([2, 2, 2])


    #my_net.train(x_train, y_train)

    #plt.plot(range(len(my_net.errors_array)), my_net.errors_array)
    #plt.show()


run()