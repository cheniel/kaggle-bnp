from utils import (
    load_data,
    log_loss,
    write_submission,
    cv_random_subsample,
    cv_kfold_cross_validation,
)
import numpy as np
import xgboost as xgb
from time import time
from datetime import datetime


def prepare_data(train_x, train_y):
    return (xgb.DMatrix(train_x, label=train_y, missing=np.nan), train_y)


def create_custom_train_boosted_trees(param, num_round):
    # returns a trainer that uses the specified parameters
    def trainer(train_x, train_y):
        return train_boosted_trees(train_x, train_y, param, num_round)
    return trainer


def train_boosted_trees(train_x, train_y, param=None, num_round=20):

    if not param:

        # setup parameters for xgboost
        # info: https://xgboost.readthedocs.org/en/latest/parameter.html

        # don't tune these
        param = {
            'objective': 'multi:softprob',
            'silent': 1,
            'num_class': 2,
        }

        # to control model complexity
        param['max_depth'] = 6  # default: 6
        param['min_child_weight'] = 1  # default: 1
        param['gamma'] = 0  # default: 0

        # add randomness to make training robust to noise
        param['colsample_bytree'] = 1  # default: 1
        param['subsample'] = 1  # default: 1

        # stepsize vs num rounds. make sure when you increase one, decrease other,
        # and vice versa
        param['eta'] = 0.3  # default: 0.3

    return xgb.train(param, train_x, num_round)


def predict_boosted_trees(bst, train_x):
    return bst.predict(train_x)[:, 1]


def _tune_parameters(train_data, test_data):
    max_depths = [6, 8, 10]
    min_child_weights = [1, 2]
    gammas = [0]
    colsample_bytrees = [0.5, 1]
    subsamples = [1]
    rounds_and_eta = [
        (200, 0.05),
        (300, 0.01),
    ]

    print('max_depths:', max_depths)
    print('min_child_weights:', min_child_weights)
    print('gammas:', gammas)
    print('colsample_bytrees:', colsample_bytrees)
    print('subsamples:', subsamples)
    print('rounds_and_eta:', rounds_and_eta)

    param = {
        'objective': 'multi:softprob',
        'silent': 1,
        'num_class': 2,
    }

    smallest_error = None
    best_param = None

    for max_depth in max_depths:
        for min_child_weight in min_child_weights:
            for gamma in gammas:
                for colsample_bytree in colsample_bytrees:
                    for subsample in subsamples:
                        for num_round, eta in rounds_and_eta:
                            param['max_depth'] = max_depth
                            param['min_child_weight'] = min_child_weight
                            param['gamma'] = gamma
                            param['colsample_bytree'] = colsample_bytree
                            param['subsample'] = subsample
                            param['eta'] = eta
                            trainer = create_custom_train_boosted_trees(
                                param, num_round
                            )

                            start = time()
                            print 'Running cross validation...'
                            error = cv_kfold_cross_validation(
                                train_data['x'],
                                train_data['y'],
                                trainer,
                                predict_boosted_trees,
                                log_loss,
                                2,
                                prepare_data
                            )
                            print '...finished running cross validation. Took {0} seconds'.format(time() - start)

                            print 'error: {}'.format(error)
                            print param

                            if not smallest_error or error < smallest_error:
                                smallest_error = error
                                best_param = param.copy()

    print 'smallest error was {}. achieved by:'.format(smallest_error)
    print best_param


def _cross_validate_single_example(train_data, test_data, param, num_round):
    trainer = create_custom_train_boosted_trees(param, num_round)

    start = time()
    print 'Running cross validation...'
    error = cv_kfold_cross_validation(
        train_data['x'],
        train_data['y'],
        trainer,
        predict_boosted_trees,
        log_loss,
        5,
        prepare_data
    )
    print '...finished running cross validation. Took {0} seconds'.format(time() - start)
    print 'error: {}'.format(error)


def _export_single_example(train_data, test_data, param, num_round):
    trainer = create_custom_train_boosted_trees(param, num_round)

    xg_train, train_y = prepare_data(train_data['x'], train_data['y'])
    xg_test, _ = prepare_data(test_data['x'], None)

    start = time()
    print 'Training model...'
    classifier = trainer(xg_train, None)
    print '...model trained. Took {0} seconds'.format(time() - start)

    start = time()
    print 'Generating training predictions...'
    y_hat_train = predict_boosted_trees(classifier, xg_train)
    print '...predictions generated. Took {0} seconds'.format(time() - start)

    start = time()
    print 'Retrieving training error...'
    error = log_loss(train_y, y_hat_train)
    print 'TRAINING ERROR: {}'.format(error)
    print '...Took {} seconds'.format(time() - start)

    start = time()
    print 'Writing train output...'
    write_submission(
        train_data['ids'], y_hat_train, 'train-boosted_trees-{}.csv'.format(datetime.now())
    )
    print '...output written. Took {0} seconds'.format(time() - start)

    start = time()
    print 'Generating testing predictions...'
    y_hat_test = predict_boosted_trees(classifier, xg_test)
    print '...predictions generated. Took {0} seconds'.format(time() - start)

    start = time()
    print 'Writing test output...'
    write_submission(
        test_data['ids'], y_hat_test, 'boosted_trees-{}.csv'.format(datetime.now())
    )
    print '...output written. Took {0} seconds'.format(time() - start)

def _write_training_prediction(train_data, param, num_round):
    trainer = create_custom_train_boosted_trees(param, num_round)
    xg_train, train_y = prepare_data(train_data['x'], train_data['y'])

    start = time()
    print 'Training model...'
    classifier = trainer(xg_train, None)
    print '...model trained. Took {0} seconds'.format(time() - start)

    start = time()
    print 'Generating training predictions...'
    y_hat_train = predict_boosted_trees(classifier, xg_train)
    print '...predictions generated. Took {0} seconds'.format(time() - start)

    start = time()
    print 'Retrieving training error...'
    error = log_loss(train_y, y_hat_train)
    print 'TRAINING ERROR: {}'.format(error)
    print '...Took {} seconds'.format(time() - start)

    start = time()
    print 'Writing training output...'
    write_submission(
        train_data['ids'], y_hat_train, 'boosted_trees-{}.csv'.format(datetime.now()), True
    )
    print '...output written. Took {0} seconds'.format(time() - start)

if __name__ == "__main__":

    start = time()
    print 'Loading data...'
    train_data, test_data = load_data()
    print '...done loading data. Took {0} seconds'.format(time() - start)

    param = {
        'objective': 'multi:softprob',
        'silent': 1,
        'num_class': 2,
    }
    # param['min_child_weight'] = 1
    param['subsample'] = 0.75
    param['eta'] = 0.01
    param['colsample_bytree'] = 0.68
    param['max_depth'] = 10
    # param['gamma'] = 0

    num_round = 1800

    _export_single_example(train_data, test_data, param, num_round)
    # _write_training_prediction(train_data, param, num_round)
    # _cross_validate_single_example(train_data, test_data, param, num_round)
    # _tune_parameters(train_data, test_data)
