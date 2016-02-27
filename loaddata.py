import numpy as np

def train_data():
    data = numpy.loadtxt(open('train.csv', 'rb'), delimiter=',', skiprows=1)
    ids = data[:,0]
    y_train = data[:,1]
    x_train = data[:,2:]
    return {'ids': ids, 'y': y_train, 'x': x_train}

def test_data():
    data = numpy.loadtxt(open('test.csv', 'rb'), delimiter=',', skiprows=1)
    ids = data[:,0]
    x_test = data[:,1:]
    return {'ids': ids, 'x': x_train}
