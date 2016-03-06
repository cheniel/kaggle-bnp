import numpy as np
import csv
import re
from sklearn.preprocessing import OneHotEncoder

LABELS_TO_REMOVE = [
    'v22'  # categorical with 23420 unique values... some sort of name I guess?
]
INDICES_TO_REMOVE = [int(label[1:]) - 1 for label in LABELS_TO_REMOVE]

CATEGORICAL_DATA_LABELS = [
    'v3', 'v22', 'v24', 'v30', 'v31', 'v47', 'v52', 'v56', 'v66', 'v71', 'v74', 'v75', 'v79', 'v91', 'v107', 'v110', 'v112', 'v113', 'v125'
]
CATEGORICAL_DATA_INDICES = [
    int(label[1:]) - 1 for label in CATEGORICAL_DATA_LABELS if label not in LABELS_TO_REMOVE
]

NON_CATEGORICAL_DATA_INDICES = [
    i for i in xrange(0, 131) if i not in CATEGORICAL_DATA_INDICES and i not in INDICES_TO_REMOVE
]


def log_loss(y, p):
    y = y.astype(np.float64)
    p = p.astype(np.float64)
    return - np.average(y * np.log(p) + (1 - y) * np.log(1 - p))


def load_data():
    train_data = _load_from_csv('train.csv')
    test_data = _load_from_csv('test.csv')

    ids_train = train_data[:, 0]
    y_train = train_data[:, 1]
    x_train = train_data[:, 2:]

    ids_test = test_data[:, 0]
    x_test = test_data[:, 1:]

    x_train, x_test = _convert_categorical_data(x_train, x_test)

    return (
        {'ids': ids_train, 'y': y_train, 'x': x_train, },
        {'ids': ids_test, 'x': x_test, },
    )


def write_submission(ids, yprob, filename):
    out = open(filename, 'w')
    out.write('Id,PredictedProb\n')

    if len(ids) != len(yprob):
        print 'Ids and yprob need to be the same length.'
        return False

    for idx in range(len(ids)):
        out.write('{0},{1}\n'.format(int(ids[idx]), yprob[idx]))

    out.close()
    return True


def _load_from_csv(filename):
    raw = list(csv.reader(open(filename, 'rb'), delimiter=','))
    # Apply the transform row function to each row, except the first
    processed = map(_transform_row, raw[1:])
    # Should be ready for numpy manipulation
    return np.array(processed)


def _letter_combo_to_num(lc):
    result = 0
    num_digits = len(lc)
    idx = num_digits - 1
    while idx > -1:
        digit = num_digits - idx
        result += (ord(lc[idx]) - ord('A') + 1) * 26**(digit - 1)
        idx -= 1
    return result


def _transform_row(row):

    def transform(item):
        if item:
            # Transform things of the form 'AA-ZZ' to integers
            if re.search('[a-zA-Z]', item):
                return float(_letter_combo_to_num(item))
            try:
                return float(item)
            except:
                return None

    # Map the row through the transform function
    return [transform(item) for item in row]


def _convert_categorical_data(x_train, x_test):
    x_train_rows = x_train.shape[0]

    combined_x = np.vstack((x_train, x_test,))

    # get all of the categorical features
    categorical_x = combined_x[:, CATEGORICAL_DATA_INDICES]

    # get all of the non-categorical features
    non_categorical_x = combined_x[:, NON_CATEGORICAL_DATA_INDICES]

    # convert 'None' to 0.0 in categorical data
    categorical_x = np.where(
        categorical_x == np.array(None), 0.0, categorical_x
    )

    # convert the categorical features using one hot encoding
    enc = OneHotEncoder(sparse=False, dtype=np.float)
    categorical_x = enc.fit_transform(categorical_x)

    # hstack the non-categorical with the converted categorical to get new x
    combined_x = np.hstack((non_categorical_x, categorical_x))

    x_train = combined_x[:x_train_rows, :]
    x_test = combined_x[x_train_rows:, :]

    return (x_train, x_test)
