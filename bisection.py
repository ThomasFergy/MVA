import numpy as np

class bisection:

    def __init__(self, function, range):
        self.f = function
        self.x_min = range[0]
        self.x_max = range[1]

    def find_max(self, num = 1000000):
        f = self.f
        a = self.x_min
        b = self.x_max
        for _ in range(num):
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
            if b-a < 0.00000000000001:
                print('b-a condition reached')
                break
        return m


