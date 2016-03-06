from utils import load_data, log_loss, write_submission
import numpy as np
import xgboost as xgb
from time import time
from datetime import datetime

start = time()
print 'Loading data...'
train_data, test_data = load_data()
print '...done loading data. Took {0} seconds'.format(time() - start)

# todo: consider adding weights. esp re: 50
# https://www.kaggle.com/bobcz3/bnp-paribas-cardif-claims-management/exploring-bnp-data-distributions

start = time()
print 'Loading DMatrix...'
xg_train = xgb.DMatrix(train_data['x'], label=train_data['y'], missing=np.nan)
xg_test = xgb.DMatrix(test_data['x'], missing=np.nan)
print '...done loading DMatrix. Took {0} seconds'.format(time() - start)


# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softprob'
# scale weight of positive examples
param['eta'] = 0.05
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 2

watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
num_round = 300

start = time()
print 'Training model...'
# bst = xgb.train(param, xg_train, num_round, watchlist )
bst = xgb.train(param, xg_train, num_round)
print '...model trained. Took {0} seconds'.format(time() - start)

start = time()
print 'Retrieving training error...'
error = log_loss(train_data['y'], bst.predict( xg_train )[:, 1])
print 'TRAINING ERROR: {}'.format(error)
print '...Took {} seconds'.format(time() - start)

# do the same thing again, but output probabilities
# Note: this convention has been changed since xgboost-unity
# get prediction, this is in 1D array, need reshape to (ndata, nclass)
start = time()
print 'Generating predictions on testing date...'
yprob = bst.predict( xg_test )  # .reshape( test_['y'].shape[0], 6 )
ylabel = np.argmax(yprob, axis=1)
print '...predictions generated. Took {0} seconds'.format(time() - start)

start = time()
print 'Writing output...'
write_submission(
    test_data['ids'], yprob[:, 1], 'boosted_trees-{}.csv'.format(datetime.now())
)
print '...output written. Took {0} seconds'.format(time() - start)


# print ('predicting, classification error=%f' % (sum( int(ylabel[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y)) ))
