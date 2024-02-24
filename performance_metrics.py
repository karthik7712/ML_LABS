import math
import numpy as np


class PerformanceMetrics:
    def __init__(self,x,y):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        for i in range(0,len(x)):
            if x[i] == -1 and y[i] == -1:
                self.tp += 1
            elif x[i] == -1 and y[i] == 1:
                self.fn += 1
            elif x[i] == 1 and y[i] == -1:
                self.fp += 1
            else:
                self.tn += 1

        self.conf_dictionary = np.array([[self.tp,self.fn],[self.fp,self.tn]])

    def confusion_matrix(self):
        return self.conf_dictionary

    def accuracy(self):
        true_preds = self.tp + self.tn
        total_vals = self.conf_dictionary.sum()
        return true_preds/total_vals

    def precision(self):
        return self.tp /(self.tp + self.fp)

    def recall(self):
        return self.tp /(self.tp + self.fn)

    def true_positive_rate(self):
        return self.tp / (self.tp + self.fn)

    def false_alarming_rate(self):
        return self.fp / (self.fp + self.tn)

    def gmean(self):
        product = self.recall() * (1-self.recall())
        return math.sqrt(product)

    def fmeasure(self):
        return (2*self.precision()*self.recall())/(self.precision()+self.recall())

    def area_under_roc(self):
        return (self.true_positive_rate()+self.false_alarming_rate()-1)/2


