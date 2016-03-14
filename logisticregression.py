from utils import load_data, log_loss, write_submission, cv_kfold_cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer
import time
from datetime import datetime
from scipy.optimize import minimize
import numpy as np

test_imputing_methods = False
test_regularization_param = False
produce_testing_data = True

BEST_C = 0.9296875

start = time.time()
print 'Loading data...'
train_data, test_data = load_data()
print '...done loading data. Took {0} seconds'.format(time.time() - start)

def imputing_test(train_data):
    # model = train_func(x_train, y_train)
    def train_func(x_train, y_train):
        print 'Training...'
        start = time.time()
        classifier = LogisticRegression(C=BEST_C, solver='liblinear', n_jobs=-1)
        classifier.fit(x_train, y_train.tolist())
        print '...done. Took {0} seconds'.format(time.time() - start)
        return classifier

    # y_hat = predict_func(model, x_test)
    def predict_func(model, x_test):
        print 'Predicting...'
        start = time.time()
        y_hat = model.predict_proba(x_test)
        print '...done. Took {0} seconds'.format(time.time() - start)
        return y_hat[:, 1]

    # error = eval_func(y_test, y_hat)
    def eval_func(y_test, y_hat):
        print 'Calculating error...'
        start = time.time()
        error = log_loss(y_test, y_hat)
        print '...done. Took {0} seconds'.format(time.time() - start)
        return error

    x = train_data['x']
    y = train_data['y']

    results = {}

    for opt in ['most_frequent', 'mean', 'median']:
        print 'Testing {0}'.format(opt)
        i = Imputer(strategy=opt)
        this_x = i.fit_transform(x)
        print '...done imputing.'
        results[opt] = cv_kfold_cross_validation(this_x, y, train_func, predict_func, eval_func, 5)
        print 'Result for opt {0} was {1}'.format(opt, results[opt])

    return results

if test_imputing_methods:
    print imputing_test(train_data)
    quit()

imputer = Imputer(strategy='mean')

print 'Imputing mmissing X values in training data...'
start = time.time()
train_data['x'] = imputer.fit_transform(train_data['x'])
print '...done imputing. Took {0} seconds'.format(time.time() - start)

print 'Imputing missing X values in testing data...'
start = time.time()
test_data['x'] = imputer.fit_transform(test_data['x'])
print '...done imputing. Took {0} seconds'.format(time.time() - start)

def param_tuning(train_data):
    # y_hat = predict_func(model, x_test)
    def predict_func(model, x_test):
        print 'Predicting...'
        start = time.time()
        y_hat = model.predict_proba(x_test)
        print '...done. Took {0} seconds'.format(time.time() - start)
        return y_hat[:, 1]

    # error = eval_func(y_test, y_hat)
    def eval_func(y_test, y_hat):
        print 'Calculating error...'
        start = time.time()
        error = log_loss(y_test, y_hat)
        print '...done. Took {0} seconds'.format(time.time() - start)
        return error

    c_dict = {}

    def objective_function(c):
        if not (0 < c[0] <= 1):
            return float('inf')

        # model = train_func(x_train, y_train)
        def train_func(x_train, y_train):
            print 'Training...'
            start = time.time()
            classifier = LogisticRegression(C=c[0], solver='liblinear', n_jobs=-1)
            classifier.fit(x_train, y_train.tolist())
            print '...done. Took {0} seconds'.format(time.time() - start)
            return classifier

        print 'Trying with C: {}'.format(c)
        error = cv_kfold_cross_validation(train_data['x'], train_data['y'], train_func, predict_func, eval_func, 5)
        print 'Got error: {}'.format(error)
        c_dict[c[0]] = error
        return error

    C0 = [.5]
    results = minimize(objective_function, C0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
    best_c = results['x']
    print 'Best C is {0}'.format(best_c)
    print c_dict

if test_regularization_param:
    param_tuning(train_data)
    quit()

def output_test_y_hat(train_data, test_data):
    print 'Training...'
    start = time.time()
    model = LogisticRegression(C=BEST_C, solver='liblinear', n_jobs=-1)
    model.fit(train_data['x'], train_data['y'].tolist())
    print '...done. Took {0} seconds'.format(time.time() - start)

    print 'Predicting...'
    start = time.time()
    y_hat = model.predict_proba(test_data['x'])
    print '...done. Took {0} seconds'.format(time.time() - start)

    print 'Writing output...'
    start = time.time()
    write_submission(test_data['ids'], y_hat[:,1], 'logistic-regression-{}.csv'.format(datetime.now()), False)
    print '...done. Took {0} seconds'.format(time.time() - start)

if produce_testing_data:
    output_test_y_hat(train_data, test_data)
    quit()
