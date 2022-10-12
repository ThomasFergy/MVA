from random import random
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

class monte_carlo:

    def __init__(self, f, x_min, x_max, para=None):
        self.f = f
        self.min = x_min
        self.max = x_max
        self.para = para
        
    def random_number_generator(self, minimum, maximum):
        num = np.random.rand()
        return minimum + ((maximum - minimum) * num)
    
    def p_max(self):
        max_x = opt.fmin(lambda x: -self.f(x, *self.para), 0, disp=False)
        p = self.f(max_x, *self.para)
        return p
    
    def mc_rejection(self, num=1000):
        pmax = self.p_max()
        random_numbers = np.zeros(num)

        i = 0
        while i < num:
            random_x = self.random_number_generator(self.min, self.max)
            p_of_x = self.f(random_x, *self.para)
            random_c = self.random_number_generator(0, pmax)

            if p_of_x > random_c:
                random_numbers[i] = random_x
                i+=1

        return random_numbers




def gauss(x, mu, sigma):
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-(1/2)*((x-mu)/sigma)**2)


class monte_carlo_superposition:

        def __init__(self, functions_and_parameters):

            self.functions = np.array(functions_and_parameters)[0:, 0]
            self.parameters = np.array(functions_and_parameters)[0:, 1]


        def superimpose(self, x_range, num, x=True):
            x_min = x_range[0]
            x_max = x_range[1]
            monte_carlos = [monte_carlo(self.functions[i], x_min, x_max, *[self.parameters[i]]) for i in range(len(self.functions))]
            x_mc = [monte_carlos[i].mc_rejection(num) for i in range(len(monte_carlos))]
            if x:
                return [x_mc, [item for sublist in x_mc for item in sublist]]
            return x_mc
            



def test(x):
    return -(x-5.6349257823794)**2 + x

class bisection:

    def __init__(self, function, range):
        self.f = function
        self.x_min = range[0]
        self.x_max = range[1]

    def find_max(self, num = 1000000):
        f = self.f
        a = self.x_min
        b = self.x_max
        i = 0
        while i < num:
            m = (a + b) / 2
            l = (a + m) / 2
            r = (b + m) / 2

            f_x = [f(a), f(l), f(m), f(r), f(b)]
            index = np.argmax(f_x)
            if index == 0 or index == 1:
                b = m
            if index == 3 or index == 4:
                a = m
            if index == 2:
                a = l
                b = r
            if b-a < 0.000000000001:
                print('b-a condition reached')
                break
            i += 1
        return m





class significance:

    def __init__(self, signal, background):
        self.x = None
        self.y = None
        self.cuts = None
        self.sig = None
        self.signal = signal
        self.background = background


    def larger(self, cut_values):
        sig = []
        self.cuts = cut_values
        for cut in cut_values:
            s = []
            b = []
            for sx in self.signal:
                if sx >= cut:
                    s.append(sx)
            for bx in self.background:
                if bx >= cut:
                    b.append(bx)
            sig.append(np.sum(s) / np.sqrt(np.sum(b)))
        
        self.sig = sig
        return sig

    def fit(self):

        num = len(self.cuts)
        estimate_index = np.argmax(self.sig)
        lower = self.cuts[estimate_index-int(num/30)]
        upper = self.cuts[estimate_index+int(num/30)]

        filtered_sig, filtered_cuts = [], []

        for i, x in enumerate(self.cuts):
            if x >= lower and x <= upper:
                filtered_cuts.append(x)
                filtered_sig.append(self.sig[i])


        x1 = filtered_cuts
        y1 = filtered_sig
        f = np.poly1d( np.polyfit(x1, y1, 50) )
        x = np.linspace(lower, upper, 10000)
        plt.plot(x, f(x), '--r')
        plt.plot(x1, y1, '.k')
        #plt.show()

        b = bisection(f, [lower, upper])
        m = b.find_max()
        print('cut =', m)
        



import time
start_time = time.time()



if __name__ == "__main__":

    monte_carlo_conditions = [[gauss, [10, 5]], [gauss, [15, 2]]]
    s = monte_carlo_superposition(monte_carlo_conditions)
    x_values = s.superimpose([-5, 25], 10000, x=False)

    #plt.hist(x_values[1], bins=500)

    x_cuts = np.linspace(-3, 20, 1000)

    sig = significance(x_values[1], x_values[0])    

    s = sig.larger(x_cuts)
    sig.fit()

  


    print("--- %s seconds ---" % (time.time() - start_time))
    plt.show()

  


