def writesubmission(ids, y_hat, filename):
    out = open(filename, 'w')
    out.write('Id,PredictedProb\n')
    
    if len(ids) != len(y_hat):
        print 'Ids and y_hat need to be the same length.'
        return False

    for idx in range(len(ids)):
        out.write('{0},{1}\n'.format(ids[idx], y_hat[idx]))

    out.close()
    return True