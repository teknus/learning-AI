import math

def mean(numbers):
	return sum(numbers)/float(len(numbers))

def calculateProbability(num, mean, stdev):
    '''
        prob
    '''
    exponent = math.exp(-(math.pow(num-mean, 2)/(2*math.pow(stdev, 2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

class GaussianNB:
    def fit(self, X_train, Y_train):
        self.X = X_train
        self.Y = Y_train