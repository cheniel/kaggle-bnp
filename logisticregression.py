from utils import load_data, log_loss, write_submission, cv_kfold_cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer
import time
from datetime import datetime
from scipy.optimize import minimize
import numpy as np

output_y_hat_train = False
output_y_hat_test = True

imputer = Imputer()

start = time.time()
print 'Loading data...'
train_data, test_data = load_data()
print '...done loading data. Took {0} seconds'.format(time.time() - start)

print 'Imputing mmissing X values in training data...'
start = time.time()
train_data['x'] = imputer.fit_transform(train_data['x'])
print '...done imputing. Took {0} seconds'.format(time.time() - start)

print 'Imputing missing X values in testing data...'
start = time.time()
test_data['x'] = imputer.fit_transform(test_data['x'])
print '...done imputing. Took {0} seconds'.format(time.time() - start)

def param_tuning(train_data, test_data):
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

    C0 = [1.0]
    results = minimize(objective_function, C0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
    best_c = results['x']
    print 'Best C is {0}'.format(best_c)
    print c_dict

param_tuning(train_data, test_data)
