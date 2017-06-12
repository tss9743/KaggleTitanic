import seaborn as sns
import matplotlib.pyplot as plt
from data_handler import DataHandler


class Visualizer(object):
    def __init__(self, file_path):
        data_handler = DataHandler(file_path)
        data_handler.remove_nans()
        data_handler.categorical_to_num()
        self.df = data_handler.data

    def peaks_and_avg(self, feature):
        # peaks for survived/not survived passengers by feature
        facet = sns.FacetGrid(self.df, hue="Survived", aspect=4)
        facet.map(sns.kdeplot, feature, shade= True)
        facet.set(xlim=(0, self.df[feature].max()))
        facet.add_legend()

        # average survived passengers by feature
        fig, axis1 = plt.subplots(1, 1, figsize=(18,4))
        average_feature = self.df[[feature, "Survived"]].groupby([feature], as_index=False).mean()
        sns.barplot(x=feature, y='Survived', data=average_feature)

        plt.show()


if __name__ == '__main__':
    vis = Visualizer('./data/train.csv')
    vis.peaks_and_avg("Age")
    vis.peaks_and_avg("Pclass")