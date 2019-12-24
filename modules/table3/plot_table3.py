import numpy as np
import matplotlib.pyplot as plt
from modules.table3.get_cc import Pfpr, Ptpr
from matplotlib.font_manager import FontProperties
from sklearn.metrics import auc

class Table3:
    def __init__(self):
        pass

    def run(self, name='Hong', exist=1):
        if not exist:
            pass
        dx = 0.001
        auc_list = self.get_auc_list(dx)

        row_label = ['16x16', '32x32', '64x64', '128x128']
        # 每个 digit 需要 13 个 bit
        # 1040 * 13 = 13520
        # 272 * 13  = 3536
        # 80 * 13   = 1040
        # 32 * 13   = 416
        hash_length_list = ['1040 digits (13520 bits)', '272 digits (3536 bits)', '80 digits (1040 bits)', '32 digits (416 bits)']
        time_list = ['15.0', '14.0', '13.7', '13.1']
        self.plot(row_label, auc_list, hash_length_list, time_list, dx)

    def get_auc_list(self, dx=0.001):
        t_list = np.arange(-0.1, 1.01, dx)
        auc_list = []
        for blocksize in [16, 32, 64, 128]:
            x = Pfpr().run(blocksize, t_list)
            y = Ptpr().run(blocksize, t_list)
            result = str(auc(x, y))[:6]
            auc_list.append(result)

        return auc_list

    def plot(self, row_label, auc_list, hash_length_list, time_list, dx=0.001):
        plt.figure('Table 3 Performance Comparisons under Different Block Sizes')
        plt.title('Performance Comparisons under Different Block Sizes', fontweight="bold")
        plt.axis('off')
        col_label = ['Block size', 'AUC', 'Hash length', 'Time (s)']
        celltext = np.transpose([row_label, auc_list, hash_length_list, time_list])
        tab = plt.table(cellText=celltext, cellLoc='center', colLabels=col_label, colLoc='center', colWidths=[0.2, 0.2, 0.4, 0.2], loc='center')
        for key, cell in tab.get_celld().items():
            row, col = key
            if row == 0:
                cell.set_text_props(fontproperties=FontProperties(weight='bold'))

        plt.text(0, 0.3, 'Note: each digit takes up 13 bits')
        plt.show()
