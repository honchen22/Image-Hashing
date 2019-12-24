import matplotlib.pyplot as plt
import numpy as np
import json

'''
class TestDataset:
    import numpy as np
    """
    cc => correlation_coefficient
    """
    x = np.array([30, 40, 50, 60, 70, 80, 90, 100])
    cc_airplane = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    cc_baboon = np.array([0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.8])
    cc_house = np.array([0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.72])
    cc_lena = np.array([0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.6])
'''

class Figure2:
    def __init__(self):
        pass

    def run(self):
        with open('modules/figure2/standard_benchmark_cc_dict.json', 'r') as f:
            cc_dict = json.loads(f.read())

        self.cc_airplane = cc_dict['Airplane']
        self.cc_baboon = cc_dict['Baboon']
        self.cc_house = cc_dict['House']
        self.cc_lena = cc_dict['Lena']

        self.plot_jpeg()
        self.plot_image_scaling()
        self.plot_salt_and_pepper_noise()
        self.plot_brightness_adjustment()
        self.plot_contrast_adjustment()
        self.plot_gamma_correction()
        self.plot_gaussian_filtering()
        self.plot_image_rotation()

    def plot_jpeg(self):
        attack_type='JPEG'
        x = self.cc_airplane[attack_type]['param_range']
        y1 = self.cc_airplane[attack_type]['cc_list']
        y2 = self.cc_baboon[attack_type]['cc_list']
        y3 = self.cc_house[attack_type]['cc_list']
        y4 = self.cc_lena[attack_type]['cc_list']
        self.plot(x, y1, y2, y3, y4, title='(a) JPEG Compression', x_label='Quality factor')

    def plot_watermark_embedding(self, x, y1, y2, y3, y4):
        self.plot(x, y1, y2, y3, y4, title='(b) Watermark embedding', x_label='Strength')

    def plot_image_scaling(self):
        attack_type='Image_Scaling'
        x = self.cc_airplane[attack_type]['param_range']
        y1 = self.cc_airplane[attack_type]['cc_list']
        y2 = self.cc_baboon[attack_type]['cc_list']
        y3 = self.cc_house[attack_type]['cc_list']
        y4 = self.cc_lena[attack_type]['cc_list']
        self.plot(x, y1, y2, y3, y4, title='(c) Image scaling', x_label='Ratio')

    def plot_speckle_noise(self, x, y1, y2, y3, y4):
        self.plot(x, y1, y2, y3, y4, title='(d) Speckle noise', x_label='Variance')

    def plot_salt_and_pepper_noise(self):
        attack_type='Salt_and_pepper_noise'
        x = self.cc_airplane[attack_type]['param_range']
        y1 = self.cc_airplane[attack_type]['cc_list']
        y2 = self.cc_baboon[attack_type]['cc_list']
        y3 = self.cc_house[attack_type]['cc_list']
        y4 = self.cc_lena[attack_type]['cc_list']
        self.plot(x, y1, y2, y3, y4, title='(e) Salt and pepper noise', x_label='Density')

    def plot_brightness_adjustment(self):
        attack_type='Brightness'
        x = self.cc_airplane[attack_type]['param_range']
        y1 = self.cc_airplane[attack_type]['cc_list']
        y2 = self.cc_baboon[attack_type]['cc_list']
        y3 = self.cc_house[attack_type]['cc_list']
        y4 = self.cc_lena[attack_type]['cc_list']
        self.plot(x, y1, y2, y3, y4, title='(f) Brightness adjustment', x_label="Photoshop's scale")

    def plot_contrast_adjustment(self):
        attack_type='Contrast'
        x = self.cc_airplane[attack_type]['param_range']
        y1 = self.cc_airplane[attack_type]['cc_list']
        y2 = self.cc_baboon[attack_type]['cc_list']
        y3 = self.cc_house[attack_type]['cc_list']
        y4 = self.cc_lena[attack_type]['cc_list']
        self.plot(x, y1, y2, y3, y4, title='(g) Contrast adjustment', x_label="Photoshop's scale")

    def plot_gamma_correction(self):
        attack_type='Gamma'
        x = self.cc_airplane[attack_type]['param_range']
        y1 = self.cc_airplane[attack_type]['cc_list']
        y2 = self.cc_baboon[attack_type]['cc_list']
        y3 = self.cc_house[attack_type]['cc_list']
        y4 = self.cc_lena[attack_type]['cc_list']
        self.plot(x, y1, y2, y3, y4, title='(h) Gamma correction', x_label='gamma')

    def plot_gaussian_filtering(self):
        attack_type='Gaussian'
        x = self.cc_airplane[attack_type]['param_range']
        y1 = self.cc_airplane[attack_type]['cc_list']
        y2 = self.cc_baboon[attack_type]['cc_list']
        y3 = self.cc_house[attack_type]['cc_list']
        y4 = self.cc_lena[attack_type]['cc_list']
        self.plot(x, y1, y2, y3, y4, title='(i) Gaussian filtering', x_label='stdev')

    def plot_image_rotation(self):
        attack_type='Rotation'
        x = self.cc_airplane[attack_type]['param_range']
        y1 = self.cc_airplane[attack_type]['cc_list']
        y2 = self.cc_baboon[attack_type]['cc_list']
        y3 = self.cc_house[attack_type]['cc_list']
        y4 = self.cc_lena[attack_type]['cc_list']
        self.plot(x, y1, y2, y3, y4, title='(j) Image rotation', x_label='Rotation angle')

    def global_setting(self, x, cc_airplane, cc_baboon, cc_house, cc_lena):
        plt.plot(x, cc_airplane, color='red', linestyle='-', marker='x', markersize=8)
        plt.plot(x, cc_baboon, color='#ff00ff', linestyle='-', marker='o', markerfacecolor='none')
        plt.plot(x, cc_house, color='black', linestyle='-', marker='s', markerfacecolor='none')
        plt.plot(x, cc_lena, color='blue', linestyle='-', marker='^', markerfacecolor='none', markersize=8)
        plt.legend(labels=['airplane', 'baboon', 'house', 'lena'], loc='lower right')
        plt.grid(True)

    def plot(self, x, y1, y2, y3, y4, title='', x_label=''):
        plt.figure(title)
        plt.ylabel('Correlation coefficient')
        plt.xlabel(x_label)
        plt.title(title, y=1.05)
        plt.axis(xmin=np.min(x), xmax=np.max(x), ymin=0, ymax=1.01)
        self.global_setting(x, y1, y2, y3, y4)
        plt.show()

'''
if __name__ == '__main__':
    """
    @param dt.x
    @type np.ndarray
    @comment 是用于图像处理的参数, 比如 JPEG 压缩中的 Quality factor = [30, 40, 50, 60, 70, 80, 90, 100]

    @param dt.yi
    @type np.ndarray
    @comment 是 原图i 与 处理后的 图i 之间的 correlation_coefficient(cc)
    """

    """
    dt = TestDataset()
    fig2 = Figure2()
    fig2.plot_brightness_adjustment(
        x = dt.x,
        y1 = dt.cc_airplane,
        y2 = dt.cc_baboon,
        y3 = dt.cc_house,
        y4 = dt.cc_lena)

    """
    # dp = DatasetPlot()
    f2 = Figure2()
    f2.run()
'''