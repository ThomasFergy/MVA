import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

from bisection import *


class significance:

    def __init__(self, signal, background):
        self.x = None
        self.y = None
        self.cuts = None
        self.sig = None
        self.signal = np.array(signal)
        self.background = np.array(background)

    def larger(self, cut_values):
        sig = []
        self.cuts = np.array(cut_values)
        for cut in cut_values:
            s = self.signal[self.signal >= cut]
            b = self.background[self.background >= cut]

            sig.append(len(s) / np.sqrt(len(b)))
        self.sig = sig
        return sig

    def fit(self):

        #plt.plot(self.cuts, self.sig, '.k')
        #plt.show()

        num = len(self.cuts)
        estimate_index = np.argmax(self.sig)
        lower = self.cuts[estimate_index-int(num/33)]
        upper = self.cuts[estimate_index+int(num/33)]

        filtered_sig, filtered_cuts = [], []
        for i, x in enumerate(self.cuts):
            if x >= lower and x <= upper:
                filtered_cuts.append(x)
                filtered_sig.append(self.sig[i])

        x1 = filtered_cuts
        y1 = filtered_sig
        f = np.poly1d( np.polyfit(x1, y1, 50) )
        x = np.linspace(lower, upper, 1000)

        #plt.plot(x, f(x), '--r')
        #plt.plot(x1, y1, '.k')
        #plt.show()

        b = bisection(f, [lower, upper])
        m = b.find_max()
        print('cut =', m)
        