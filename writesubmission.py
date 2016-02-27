def writesubmission(ids, y_hat, filename):
    out = open(filename, 'w')
    out.write('Id,PredictedProb\n')
    
    if len(ids) != len(y_hat):
        print 'Ids and y_hat need to be the same length.'
        return False

    for idx in range(len(ids)):
        out.write('{0},{1}\n'.format(int(ids[idx]), y_hat[idx][1]))

    out.close()
    return True