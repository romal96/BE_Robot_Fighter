class AlphaVector(object):
    """
    Simple wrapper for the alpha vector used for representing the value function for a POMDP as a piecewise-linear,
    convex function
    """
    def __init__(self, a, v):
        self.action = a
        self.v = v

    def copy(self):
        return AlphaVector(self.action, self.v)

    def print_alpha_vect(self):
        #print("vec: ", self.v)
        #print("action: ", self.action)
        print(self.__dict__)



