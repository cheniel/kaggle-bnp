import numpy
import csv
import re

def letter_combo_to_num(lc):
    result = 0
    num_digits = len(lc)
    idx = num_digits - 1
    while idx > -1:
        digit = num_digits - idx
        result += (ord(lc[idx]) - ord('A') + 1) * 26**(digit - 1)
        idx -= 1
    return result

# Passes transform to each element of a row
def transform_row(row):
    # Transform things of the form 'AA-ZZ' to integers
    def transform(item):
        if re.search('[a-zA-Z]', item): 
            return float(letter_combo_to_num(item))
        try:
            return float(item)
        except:
            return None
    # Map the row through the transform function
    return map(transform, row)

def train_data():
    # Load the raw data in from a csv
    raw = csv.reader(open('train.csv', 'rb'), delimiter=',')
    # Apply the transform row function to each row
    processed = map(transform_row, raw)
    # Should be ready for numpy manipulation
    data = numpy.array(processed)
    ids = data[:,0]
    y_train = data[:,1]
    x_train = data[:,2:]
    return {'ids': ids, 'y': y_train, 'x': x_train}

def test_data():
    # Load the raw data in from a csv
    raw = csv.reader(open('test.csv', 'rb'), delimiter=',')
    # Apply the transform row function to each row
    processed = map(transform_row, raw)
    # Should be ready for numpy manipulation
    data = numpy.array(processed)
    ids = data[:,0]
    x_test = data[:,1:]
    return {'ids': ids, 'x': x_test}