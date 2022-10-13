from random import random
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import time
start_time = time.time()

def gauss(x, mu, sigma):
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-(1/2)*((x-mu)/sigma)**2)


class monte_carlo:

    def __init__(self, f, x_min, x_max, para=None):
        self.f = f
        self.min = x_min
        self.max = x_max
        self.para = para
        
    def random_number_generator(self, minimum, maximum):
        num = np.random.rand()
        return minimum + ((maximum - minimum) * num)
    
    #def p_max(self):
    #    max_x = opt.fmin(lambda x: -self.f(x, *self.para), 0, disp=False)
    #    p = self.f(max_x, *self.para)
    #    return p

    def p_max(self):
        return self.f(self.para[0], *self.para)
        

    
    def mc_rejection(self, num=1000):
        pmax = self.p_max()
        random_numbers = []

        i = 0
        while i < num:
            random_x = self.random_number_generator(self.min, self.max)
            p_of_x = self.f(random_x, *self.para)
            random_c = self.random_number_generator(0, pmax)

            if p_of_x > random_c:
                random_numbers.append(random_x)
                i += 1

        return random_numbers



    #def mc_rejection(self, num = 1000):
    #    pmax = self.p_max()

        

        #pass



class monte_carlo_superposition:

        def __init__(self, functions_and_parameters):
            self.functions = np.array(functions_and_parameters, dtype=object)[0:, 0]
            self.parameters = np.array(functions_and_parameters, dtype=object)[0:, 1]


        def superimpose(self, x_range, num):
            x_min = x_range[0]
            x_max = x_range[1]
            
            monte_carlos = [monte_carlo(self.functions[i], x_min, x_max, *[self.parameters[i]]) for i in range(len(self.functions))]
            #print("--- %s seconds ---" % (time.time() - start_time))
            return [monte_carlos[i].mc_rejection(num) for i in range(len(monte_carlos))]
             



if __name__ == "__main__":

    monte_carlo_conditions = [[gauss, [10, 5]], [gauss, [15, 2]]]
    s = monte_carlo_superposition(monte_carlo_conditions)
    x_values = s.superimpose([-5, 25], 10000)

    x_values = [s.superimpose([-5, 25], 1000) for i in range(100)]

    #plt.hist(x_values[0], bins=100)
    #plt.show()







    print("--- %s seconds ---" % (time.time() - start_time))