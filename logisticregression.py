import loaddata
import writesubmission
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer
import time

# Using all default values for now
classifier = LogisticRegression()
imputer = Imputer()

start = time.time()
print 'Loading data...'
train_data = loaddata.train_data()
test_data = loaddata.test_data()
print '...done loading data. Took {0} seconds'.format(time.time() - start)

print 'Imputing missing X values in training data...'
start = time.time()
train_data['x'] = imputer.fit_transform(train_data['x'])
print '...done imputing. Took {0} seconds'.format(time.time() - start)

start = time.time()
print 'Training model...'
classifier.fit(train_data['x'], train_data['y'].tolist())
print '...model trained. Took {0} seconds'.format(time.time() - start)

print 'Imputing missing X values in testing data...'
start = time.time()
test_data['x'] = imputer.fit_transform(test_data['x'])
print '...done imputing. Took {0} seconds'.format(time.time() - start)

start = time.time()
print 'Generating predictions...'
y_hat = classifier.predict_proba(test_data['x'])
print '...predictions generated. Took {0} seconds'.format(time.time() - start)

start = time.time()
print 'Writing output...'
writesubmission.writesubmission(test_data['ids'], y_hat, 'logistic-regression-output.csv')
print '...output written. Took {0} seconds'.format(time.time() - start)