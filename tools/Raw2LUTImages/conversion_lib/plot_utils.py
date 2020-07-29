import numpy as np
import matplotlib.pyplot as plt


class PlotLut(object):

    def add_plot(self, lut_kneepoints):
        b = [[0, 0]] + lut_kneepoints + [[2 ** 16, 2 ** 16]]
        print(b)
        lut_np = (1.0 * np.asarray(b)) / 2 ** 16
        print(lut_np)
        plt.plot(lut_np[:, 0], lut_np[:, 1])

    def show_plot(self):
        plt.show()
    def add_title(self, title):
        plt.title(title)