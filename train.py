import numpy as np
from data_handler import DataHandler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_val_score


class Trainer(object):
    def __init__(self, cols, classifier):
        self.cols = cols
        self.classifier = classifier

    def train(self, data):
        X = data[self.cols].values
        Y = data["Survived"].values

        if self.classifier == 'logistic':
            log_reg = LogisticRegression()
            log_reg = log_reg.fit(X,Y)
            score_log = cross_val_score(log_reg, X, Y, cv=5).mean()
            print(score_log)
        elif self.classifier == 'perceptron':
            perceptron = Perceptron(
                class_weight='balanced'
                )
            perceptron = perceptron.fit(X,Y)
            score_pctr = cross_val_score(perceptron, X, Y, cv=5).mean()
            print(score_pctr)
        else:
            raise ValueError("Unknown classifier.")


if __name__ == '__main__':
    data_handler = DataHandler('./data/train.csv')
    data_handler.remove_nans()
    data_handler.categorical_to_num()

    trainer1 = Trainer(["Age", "Sex"], 'logistic')
    trainer1.train(data_handler.data)

    trainer2 = Trainer(["Age", "Sex"], 'perceptron')
    trainer2.train(data_handler.data)