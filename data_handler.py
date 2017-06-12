import numpy as np
import pandas as pd


class DataHandler(object):
    def __init__(self, file_path):
        self.data = self._load_data(file_path)

    def _load_data(self, file_path):
        data = pd.read_csv(file_path)
        return data

    def remove_nans(self):
        # Only change Age to be mean
        mean_age = self.data["Age"].mean()
        self.data.fillna(round(mean_age), inplace=True)

    def categorical_to_num(self):
        self.data["Sex"] = self.data["Sex"].astype("category")
        self.data["Sex"].cat.categories = [0,1]
        self.data["Sex"] = self.data["Sex"].astype("int")
        self.data["Embarked"] = self.data["Embarked"].astype("category")
        self.data["Embarked"].cat.categories = [0,1,2,3]
        self.data["Embarked"] = self.data["Embarked"].astype("int")


if __name__ == '__main__':
    data_handler = DataHandler('./data/train.csv')