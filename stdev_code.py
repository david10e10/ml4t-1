#http://quantsoftware.gatech.edu/MC1-Homework-1

import math as m

def stdev_p(data):
    #result = 2.0 # your code goes here
    n = len(data)
    if n == 0:
        return None
    result = sumsquaredeviations(data)/n
    return result

# calculate the sample standard deviation
def stdev_s(data):
    #result = 1.9 # your code goes here
    n = len(data)
    if n == 1:
        result = stdev_p(data)
    elif n < 1:
        result = None
    else:
        result = sumsquaredeviations(data)/n-1

    return result

def sumsquaredeviations(data):
    n = len(data)
    sqdev = []
    if n == 0: #if an empty set, do not throw an error as the expected St.Dev = 0; should never get here, but just in case
        return 0
    avg = sum(data) / n
    for i in data:
        val = m.pow((i - avg), 2)
        sqdev.append(val)
    return sum(sqdev)

if __name__ == "__main__":
    test = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    print "the population stdev is", stdev_p(test)
    print "the sample stdev is", stdev_s(test)
