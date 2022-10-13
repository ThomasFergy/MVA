import monte_carlo as mc


import matplotlib.pyplot as plt
import numpy as np
import time

start_time = time.time()

def gauss(x, mu, sigma):
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-(1/2)*((x-mu)/sigma)**2)

if __name__ == '__main__':

    #test = mc.create(gauss, -3, 3, [0, 1])
    #x = test.mc_rejection()

    conditions = [[gauss, [10, 5]], [gauss, [15, 2]]]
    test = mc.superpose(conditions)
    x_values = [test.make([-5, 25]) for i in range(1000)]


    #print(len(x_values[0][0]))
    #print(len(x_values[0][1]))


    print("--- %s seconds ---" % (time.time() - start_time))
    #plt.show()