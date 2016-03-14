from datetime import datetime
from scipy.optimize import minimize
import numpy as np
import numpy.matlib
import os
from utils import *
import time

training = False
testing = True
# w_hat = np.array([1.26902544, -0.17764111])
w_hat = None
# w_hat = [1/float(3), 1/float(3), 1/float(3)]
w_hat = [.6, .2, .2]

# Returns a numpy array where col 1 is the ids, and col 2-n are the predictions from the models
def collect_models_from_folder():
    model_types = []
    models = []
    for output in os.listdir('bag-of-outputs'):
        models.append(read_submission('bag-of-outputs/{0}'.format(output)))
        model_types.append(output.split('-')[0])

    print models[0].shape
    models_matrix = np.zeros((models[0].shape[0], len(models) + 1))
    models_matrix[:,0] = models[0][:,0]

    idx = 1
    for m in models:
        models_matrix[:,idx] = m[:,1]
        idx += 1

    return (models_matrix, model_types)

def initialize_weights(num_models):
    val = 1 / float(num_models)
    base = np.array(val)
    return np.matlib.repmat(val, 1, num_models)

def predict_with_weights(models, weights, num_models, num_dpoints):
    shaped_weights = np.matlib.repmat(weights, num_dpoints, 1)
    weighted_models = np.multiply(models[:,1:], shaped_weights)
    predictions = np.sum(weighted_models, axis=1)
    return predictions

start = time.time()
print 'Loading models from csv output...'
(train_data, test_data) = load_data()
(models, types) = collect_models_from_folder()
num_models = models.shape[1] - 1
num_dpoints = models.shape[0]
print '...done. Took {0} seconds'.format(time.time() - start)

if training:
    for idx in range(num_models):
        print 'Training error on model {0}:'.format(types[idx])
        print log_loss(train_data['y'], models[:,idx + 1])

    start = time.time()
    print 'Weighting models and computing y_hat...'
    w0 = initialize_weights(num_models)
    y_hat = predict_with_weights(models, w0, num_models, num_dpoints)
    print '...done. Took {0} seconds'.format(time.time() - start)


    print 'Training error on ensemble model with default weights:'
    print log_loss(train_data['y'], y_hat)

    def weight_opt_predict(w):
        y_hat = predict_with_weights(models, w, num_models, num_dpoints)
        return log_loss(train_data['y'], y_hat)

    start = time.time()
    print 'Optimizing weights...'
    results = minimize(weight_opt_predict, w0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
    print results
    w_hat = results['x']

    print '...done. Took {0} seconds'.format(time.time() - start)
    print 'Final weights are {0}'.format(w_hat)

    print 'Training error with best weights:'
    print weight_opt_predict(w_hat)

if testing:
    if w_hat == None:
        w_hat = initialize_weights(num_models)

    start = time.time()
    print 'Using w_hat {0} and computing y_hat...'.format(w_hat)
    y_hat = predict_with_weights(models, w_hat, num_models, num_dpoints)
    print '...done predicting. Took {0} seconds'.format(time.time() - start)

start = time.time()
print 'Writing ensemble output...'
if training:
    write_submission(train_data['ids'], y_hat, 'ensemble-output-{}.csv'.format(datetime.now()), False)
if testing:
    write_submission(test_data['ids'], y_hat, 'ensemble-output-{}.csv'.format(datetime.now()), False)
print '...done. Took {0} seconds'.format(time.time() - start)
