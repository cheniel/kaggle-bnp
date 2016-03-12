
from lasagne.layers import InputLayer, DropoutLayer, DenseLayer
from lasagne.nonlinearities import softmax, tanh
from lasagne.nonlinearities import rectify as relu
from lasagne.updates import rmsprop, nesterov_momentum

from nolearn.lasagne import NeuralNet, PrintLayerInfo
from time import time

import theano
import numpy as np

import utils

floatX = theano.config.floatX

#================#
## LOADING DATA ##
#================#
start = time()
print 'Loading data...'
x_tr, x_vld, x_tst = utils.load_normalized_data(t_ratio=0.2)

NUM_CLASSES = 2
NUM_FEATURES = x_tr['x'].shape[1]
print '{0}'.format(NUM_FEATURES)
print 'Training set size: {0} examples'.format(x_tr['x'].shape[0])
print 'Validation set size: {0} examples'.format(x_vld['x'].shape[0])

# convert to theano types
x_train = np.asarray(x_tr['x'].astype(dtype=floatX))
x_valid = np.asarray(x_vld['x'].astype(dtype=floatX))
x_test = np.asarray(x_tst['x'].astype(dtype=floatX))
y_train = x_tr['y'].astype(dtype=np.int32)
y_valid = x_vld['y'].astype(dtype=np.int32)
end = time()
print '...done loading. Took {0} seconds'.format(end - start)

# Join training % validation sets
# x_train = np.vstack((x_train, x_valid))
# y_train = np.hstack((y_train, y_valid))

#========================#
## NETWORK ARCHITECTURE ##
#========================#
layers = [
	(InputLayer, {'shape': (None, NUM_FEATURES)}),
	# (DenseLayer, {'num_units': 100, 'nonlinearity': relu}),
	(DenseLayer, {'num_units': 75, 'nonlinearity': tanh}),
	(DropoutLayer, {'p':0.5}),
	(DenseLayer, {'num_units':NUM_CLASSES, 'nonlinearity':softmax}),
]

net = NeuralNet(
		layers=layers,
		max_epochs=100,

		# update=rmsprop,
		update=nesterov_momentum,
		update_learning_rate=0.001,
		update_momentum=0.9,

		verbose=1,
	)

net.initialize()
layer_info = PrintLayerInfo()
layer_info(net)

#====================#
## NETWORK TRAINING ##
#====================#
start = time()
print 'Training neural network...'
net.fit(x_train, y_train, epochs=60)
end = time()

m, s = divmod(end-start, 60)
print('Training runtime: {0}mins, {1}s.'.format(m,s))

#==============#
## VALIDATION ##
#==============#
y_pred = net.predict_proba(x_valid)
error = utils.log_loss(y_valid, y_pred[:,1])
print('Log loss: %.4f' % error)

#===============#
## PREDICTIONS ##
#===============#
# predictions = net.predict_proba(x_test)
# utils.write_submission(x_tst['ids'], predictions[:,1],'nn.csv')

