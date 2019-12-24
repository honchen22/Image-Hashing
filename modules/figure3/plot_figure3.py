import numpy as np
import matplotlib.pyplot as plt
import json
from modules.figure3.get_cc import CorrelationCoefficientFigure3

'''
class Dataset:
    def __init__(self):
        mu, sigma = 0, 0.1
        x = mu + sigma * np.random.randn(10000)
        print(x.shape)
        print(x[:10])
'''

class Figure3:
    def __init__(self):
        pass

    def run(self, option=None, exist=1):
        if not exist:
            cc = CorrelationCoefficientFigure3()
            cc.run()
        if not option:
            with open('modules/figure3/ucid_cc_dict_64.json', 'r') as f:
                cc_dict = json.loads(f.read())
        else:
            with open('modules/figure3/ucid_cc_dict_64_contrast_div10.json', 'r') as f:
                cc_dict = json.loads(f.read())
        
        x = cc_dict['cc_list']
        self.plot(x, option)

    def plot(self, x, option):
        plt.figure('Figure3 Distribution of correlation coefficients between two hashes')
        plt.ylabel('Frequency')
        plt.xlabel('Correlation coefficient')
        plt.title('Distribution of correlation coefficients between two hashes', y=1.05)
        if not option:
            plt.axis(xmin=0.5, xmax=1.0)
            bins = np.arange(0.5, 1, 0.001)
        else:
            plt.axis(xmin=-0.4, xmax=1.0)
            bins = np.arange(-1, 1, 0.01)
        plt.hist(x, bins=bins, facecolor='green', alpha=0.75)
        # , edgecolor="gray"
        plt.grid(True)
        plt.show()

    def table(self):
        row_label = ['T=0.9', 'T=0.8']
        col_label = ['P(FPR)[UCID]', 'P(TPR)[Copydays]']
        celltext = [[1, 2], [3, 4]]
        plt.table(cellText=celltext, rowLabels=row_label, colLabels=col_label, loc='right')
        plt.show()

'''
if __name__ == '__main__':
    """
    @param dt.x
    @type np.ndarray
    @comment    长度为 n * (n-1) / 2 = 1338 * (1338-1) / 2 = 894453, 即 1338 个 hash 值两两之间的 cc
    x 即 correlation_coefficient 相关系数
    x.shape = [1, 894453]
    x 的值域是 [-1, 1]

    @cc correlation_coefficient
    """
    # dt = Dataset()
    fig3 = Figure3()
    fig3.run()
    # fig3.plot(x = dt.x)
'''