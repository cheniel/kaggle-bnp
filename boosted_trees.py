from utils import load_data, log_loss, write_submission, cv_error
import numpy as np
import xgboost as xgb
from time import time
from datetime import datetime


def prepare_data(train_x, train_y):
    return (xgb.DMatrix(train_x, label=train_y, missing=np.nan), train_y)


def train_boosted_trees(train_x, train_y):

    # setup parameters for xgboost
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.05
    param['max_depth'] = 6
    param['silent'] = 1
    param['nthread'] = 4
    param['num_class'] = 2
    num_round = 300

    return xgb.train(param, train_x, num_round)


def predict_boosted_trees(bst, train_x):
    return bst.predict(train_x)[:, 1]

if __name__ == "__main__":

    start = time()
    print 'Loading data...'
    train_data, test_data = load_data()
    print '...done loading data. Took {0} seconds'.format(time() - start)

    print cv_error(train_data['x'], train_data['y'], train_boosted_trees, predict_boosted_trees, log_loss, prepare_data)

    # start = time()
    # print 'Loading data...'
    # train_data, test_data = load_data()
    # xg_train, train_y = prepare_data(train_data['x'], train_data['y'])
    # xg_test, _ = prepare_data(test_data['x'], None)
    # print '...done loading data. Took {0} seconds'.format(time() - start)
    #
    # start = time()
    # print 'Training model...'
    # classifier = train_boosted_trees(xg_train, None)
    # print '...model trained. Took {0} seconds'.format(time() - start)
    #
    # start = time()
    # print 'Generating training predictions...'
    # y_hat_train = predict_boosted_trees(classifier, xg_train)
    # print '...predictions generated. Took {0} seconds'.format(time() - start)
    #
    # start = time()
    # print 'Retrieving training error...'
    # error = log_loss(train_y, y_hat_train)
    # print 'TRAINING ERROR: {}'.format(error)
    # print '...Took {} seconds'.format(time() - start)
    #
    # start = time()
    # print 'Generating testing predictions...'
    # y_hat_test = predict_boosted_trees(classifier, xg_test)
    # print '...predictions generated. Took {0} seconds'.format(time() - start)
    #
    # start = time()
    # print 'Writing output...'
    # write_submission(
    #     test_data['ids'], y_hat_test, 'boosted_trees-{}.csv'.format(datetime.now())
    # )
    # print '...output written. Took {0} seconds'.format(time() - start)
