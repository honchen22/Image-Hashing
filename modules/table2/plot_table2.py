import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

class Table2:
    def __init__(self):
        pass

    def run(self, name='Hong', exist=1):
        if not exist:
            pass
        from modules.table2.get_cc import Pfpr, Ptpr
        t_list = [0.92, 0.94, 0.96, 0.97, 0.98, 0.99, 0.999]
        p_fpr_ucid = Pfpr().run(t_list)
        p_tpr_copydays = Ptpr().run(t_list)
        row_label = ['T=%s' % str(t) for t in t_list]
        self.plot(row_label, p_fpr_ucid, p_tpr_copydays)

    def plot(self, row_label, p_fpr_ucid, p_tpr_copydays):
        plt.figure('Table 2 Pfpr and Ptpr under Different Threshold')
        plt.title('Pfpr and Ptpr under Different Threshold', fontweight='bold')
        plt.axis('off')
        # row_label = ['T=0.90', 'T=0.91', 'T=0.92', 'T=0.93', 'T=0.94', 'T=0.95', 'T=0.96']
        col_label = ['Threshold', 'Pfpr of UCID', 'Ptpr of Copydays ']
        celltext = np.transpose([row_label, p_fpr_ucid, p_tpr_copydays])
        tab = plt.table(cellText=celltext, cellLoc='center', colLabels=col_label, colLoc='center', loc='center')
        for key, cell in tab.get_celld().items():
            row, col = key
            if row == 0:
                cell.set_text_props(fontproperties=FontProperties(weight='bold'))
        plt.show()

# if __name__ == '__main__':
#     """
#     当阈值为 T 时，
#     p_fpr_ucid 是 UCID 的 𝑃(FPR), 即 Figure3(UCID) 中 cc > T 的比例
#     p_tpr_copydays 是 Copydays  的 𝑃(TPR), 
#     """
#     # p_fpr_ucid = [0, 1, 2, 3, 4, 5, 6]
#     # p_tpr_copydays = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
#     # table2 = Table2()
#     # table2.plot(p_fpr_ucid, p_tpr_copydays)
#     t = Table2()
#     t.run()
