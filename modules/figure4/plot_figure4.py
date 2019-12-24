import numpy as np
import matplotlib.pyplot as plt
from modules.figure4.get_cc import Pfpr, Ptpr, CopydaysHash2CCParser, UCIDHash2CCParser

class Figure4:
    def __init__(self):
        self.t_list = [0.8, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999, 0.9991, 0.9992, 0.9993, 0.9994, 0.9996, 0.9999, 1]
        # self.t_list = np.arange(-0.1, 1.01, 0.01)
        self.blocksize_16 = 16
        self.blocksize_32 = 32
        self.blocksize_64 = 64
        self.blocksize_128 = 128

    def run(self, exist=1):
        if not exist:
            pass

            # parser = CopydaysHash2CCParser()
            # parser.run(self.blocksize_16)
            # parser.run(self.blocksize_32)
            # parser.run(self.blocksize_64)
            # parser.run(self.blocksize_128)

            # parser = UCIDHash2CCParser()
            # parser.run(self.blocksize_16)
            # print('x16 done')
            # parser.run(self.blocksize_32)
            # print('x32 done')
            # parser.run(self.blocksize_64)
            # print('x64 done')
            # parser.run(self.blocksize_128)
            # print('x128 done')

        plt.figure('Receiver Operating Characteristics(ROC)')
        plt.axis(xmin=-0.005, xmax=1.0, ymin=0, ymax=1.0)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.grid(True)

        self.plot_x_y_16()
        self.plot_x_y_32()
        self.plot_x_y_64()
        self.plot_x_y_128()

        # plt.legend(labels=['64x64'], loc='lower right')
        plt.legend(labels=['16x16', '32x32', '64x64', '128x128'], loc='lower right')
        plt.show()


    def plot_x_y_16(self):
        x = Pfpr().run(self.blocksize_16, self.t_list)
        y = Ptpr().run(self.blocksize_16, self.t_list)
        plt.plot(x, y, color='#ff00ff', marker='o', markerfacecolor='none')

    def plot_x_y_32(self):
        x = Pfpr().run(self.blocksize_32, self.t_list)
        y = Ptpr().run(self.blocksize_32, self.t_list)
        plt.plot(x, y, color='black', marker='*')

    def plot_x_y_64(self):
        x = Pfpr().run(self.blocksize_64, self.t_list)
        y = Ptpr().run(self.blocksize_64, self.t_list)
        plt.plot(x, y, color='red', linewidth=1.2, marker='x')

    def plot_x_y_128(self):
        x = Pfpr().run(self.blocksize_128, self.t_list)
        y = Ptpr().run(self.blocksize_128, self.t_list)
        plt.plot(x, y, color='blue', marker='^')
