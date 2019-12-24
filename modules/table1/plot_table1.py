import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import json

class Table1:
    def __init__(self):
        pass

    def run(self, name='Hong', exist=1):
        if not exist:
            from modules.table1.get_cc import CorrelationCoefficientTable1
            cc = CorrelationCoefficientTable1()
            cc.run(name)
        with open('modules/table1/copydays_cc_dict_64.json', 'r') as f:
            cc_dict = json.loads(f.read())

        row_label, maximum, minimum, mean = [], [], [], []
        for attack_type in cc_dict:
            row_label.append(attack_type)
            maximum.append(str(cc_dict[attack_type]['maximum'])[:6])
            minimum.append(str(cc_dict[attack_type]['minimum'])[:6])
            mean.append(str(cc_dict[attack_type]['mean'])[:6])

        self.plot(row_label, maximum, minimum, mean)

    def plot(self, row_label, maximum, minimum, mean):
        plt.figure('Table 1 Statistics of Correlation Coefficient under Different Operations')
        ttl = plt.title('Statistics of Correlation Coefficient under Different Operations', fontweight="bold")
        plt.axis('off')
        # row_label = ['JPEG compression', 'Watermark embedding', 'Image scaling', 'Speckle noise', 'Salt and Pepper noise', 'Brightness adjustment', 'Contrast adjustment', 'Gamma correction', 'Gaussian filtering', 'Image rotation']
        col_label = ['Operation', 'Maximum', 'Minimum ', 'Mean']
        celltext = np.transpose([row_label, maximum, minimum, mean])
        tab = plt.table(cellText=celltext, cellLoc='center', colLabels=col_label, colLoc='center', colWidths=[0.4, 0.2, 0.2, 0.2], loc='center')
        for key, cell in tab.get_celld().items():
            row, col = key
            if row == 0:
                cell.set_text_props(fontproperties=FontProperties(weight='bold'))
        plt.show()

# Table1().run('Hong')