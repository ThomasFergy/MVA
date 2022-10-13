import numpy as np

class create:

    def __init__(self, f, x_min, x_max, para=None):
        self.f = f
        self.min = x_min
        self.max = x_max
        self.para = para

    def p_max(self):    # Only works for Gauss
        return self.f(self.para[0], *self.para)
    
    def mc_rejection(self, num=1000):
        N = num*((self.max - self.min) // 2)
        a = self.min
        b = self.max
        m = self.p_max()
        u1 = np.random.uniform(a, b, N)
        u2 = np.random.uniform(0.0, m, N)
        rej = np.empty(N)
        rej.fill(0)
        variables = np.where(u2 <= self.f(u1, *self.para), u1, rej)
        accept = np.extract(variables != 0, variables)
        return accept[:num]


class superpose:

        def __init__(self, functions_and_parameters):
            self.functions = np.array(functions_and_parameters, dtype=object)[0:, 0]
            self.parameters = np.array(functions_and_parameters, dtype=object)[0:, 1]


        def make(self, x_range, num=1000):
            x_min = x_range[0]
            x_max = x_range[1]
            
            monte_carlos = [create(self.functions[i], x_min, x_max, *[self.parameters[i]]) for i in range(len(self.functions))]
            return [monte_carlos[i].mc_rejection(num) for i in range(len(monte_carlos))]