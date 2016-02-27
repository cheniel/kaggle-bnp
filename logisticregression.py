import loaddata
import writesubmission
from sklearn.linear_model import LogisticRegression
import time

# Using all default values for now
classifier = LogisticRegression()

start = time.time()
print 'Loading data...'
train_data = loaddata.train_data()
test_data = loaddata.test_data()
print '...done. Took {0} seconds'.format(time.time() - start)

classifier.fit(train_data['x'], train_data['y'])
y_hat = classifier.predict_proba(test_data['x'])

writesubmission.writesubmission(test_data['ids'], y_hat, 'logistic-regression-output.csv')
