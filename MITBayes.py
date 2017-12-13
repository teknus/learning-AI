import math


def calculateProbability(num, mean, stdev):
    exponent = math.exp(-(math.pow(num-mean, 2)/(2*math.pow(stdev, 2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

def mean(numbers):
	return sum(numbers)/float(len(numbers))

def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries

class GaussianNB:
    def fit(self, X_train, Y_train):
        self.X = X_train
        self.Y = Y_train
        self.sepClass()

    def predict(self,X_test):
        predictions = []
        for i in range(len(X_test)):
            result = self.getpredict(X_test[i])
            predictions.append(result)
        return predictions

    def getpredict(self, X_test):
        probabilities = self.calculateClassProbabilities(X_test)
        bestLabel, bestProb = None, -1
        for classValue, probability in probabilities.items():
            if bestLabel is None or probability > bestProb:
                bestProb = probability
                bestLabel = classValue
        return bestLabel

    def summarizeByClass(self):
        summaries = {}
        for classValue, instances in self.sep.items():
            summaries[classValue] = summarize(instances)
        return summaries

    def sepClass(self):
        self.sep = {}
        for y in self.Y:
            self.sep[y] = []
        for label,point in zip(self.Y,self.X):
            self.sep[label].append(list(point))
        self.summaries = self.summarizeByClass()
    
    def calculateClassProbabilities(self, inputVector):
        probabilities = {}
        for classValue, classSummaries in self.summaries.items():
            probabilities[classValue] = 1
            for i in range(len(classSummaries)):
                mean, stdev = classSummaries[i]
                x = inputVector[i]
                probabilities[classValue] *= calculateProbability(x, mean, stdev)
        return probabilities