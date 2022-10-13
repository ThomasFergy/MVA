import monte_carlo as mc
import significance as s


import matplotlib.pyplot as plt
import numpy as np
import time

start_time = time.time()

def gauss(x, mu, sigma):
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-(1/2)*((x-mu)/sigma)**2)

if __name__ == '__main__':

    conditions = [[gauss, [15, 2]], [gauss, [10, 5]]]
    test = mc.superpose(conditions)
    x_values = [test.make([0, 20], 1000) for i in range(999)]

    x = test.make([10, 20], 1000)

    significance = s.significance(x[0], x[1])

    x_cuts = np.linspace(10 , 15, 1000)
    significance.larger(x_cuts)
    significance.fit()



    print("--- %s seconds ---" % (time.time() - start_time))
    #plt.show()