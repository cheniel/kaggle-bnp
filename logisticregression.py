import loaddata
import writesubmission
import sklearn

# Using all default values for now
classifier = sklearn.linear_model.LogisticRegression()

train_data = loaddata.train_data()
test_data = loaddata.test_data()

classifier.fit(train_data['x'], train_data['y'])
y_hat = classifier.predict_proba(test_data['x'])

writesubmission.writesubmission(test_data['ids'], y_hat, 'logistic-regression-output.csv')
