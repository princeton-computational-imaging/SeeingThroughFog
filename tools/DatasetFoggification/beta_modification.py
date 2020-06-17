import numpy as np


class BetaRadomization():

    def __init__(self, beta):
        """
        Do initliatization

        """


        self.mhf = 2 # maximal horzontal frequency
        self.mvf = 5 # maximal vertical frequency
        self.height_max = 5
        self.offset = []


        self.beta = beta

        # sample number of furier components, sample random offsets to one another, # Independence Height and angle
        self.number_height = np.random.randint(3,5)
        self.number_angle = np.random.randint(6,10)

        # sample frequencies
        self.frequencies_angle = np.random.randint(1, self.mhf, size=self.number_angle)
        self.frequencies_height = np.random.randint(0, self.mvf, size=self.number_angle)
        # sample frequencies
        self.offseta = np.random.uniform(0, 2*np.pi, size=self.number_angle)
        self.offseth = np.random.uniform(0, 2*np.pi, size=self.number_angle)
        self.intensitya = np.random.uniform(0, 0.1/self.number_angle/2, size=self.number_angle)
        self.intensityh = np.random.uniform(0, 0.1/self.number_angle/2, size=self.number_angle)

        pass

    def propagate_in_time(self, timestep):
        self.offseta += self.frequencies_angle * timestep/10
        self.offseth += self.frequencies_height * timestep / 10
        pass

    def setup(self, beta):
        pass

    def _function(self, angle_h=None, height=None):
        was_None = False
        if height is None:
            height = np.linspace(0, self.height_max, 200)/self.height_max*2*np.pi
            was_None = True

        if angle_h is None:
            angle_h = np.linspace(0, 2*np.pi, 200)
            was_None = True
        a = 0
        h = 0
        if was_None:
            a, h = np.meshgrid(angle_h, height)
        else:
            a = angle_h
            h = height

        output = np.zeros(np.shape(a))
        for fa, fh, oa, oh, Ah, Aa in zip(self.frequencies_angle, self.frequencies_height, self.offseta, self.offseth, self.intensityh, self.intensitya):
            output += np.abs((Aa*np.sin(fa*a+oa)/fa+Ah*np.sin(fa*a+fh*h+oh)))

        output += self.beta
        print(output)
        return output

    def _print_function(self):
        """
        Print function for values inbetween 0-360 and inbetween different heights
        :return:
        """
        pass




    def get_beta(self, distance_forward, right, height):
        distance_forward = np.where(distance_forward == 0, np.ones_like(distance_forward) * 0.0001, distance_forward)
        angle = np.tan(np.divide(right, distance_forward))
        beta_usefull = self._function(angle, height)

        return beta_usefull



if __name__ == '__main__':
    C = BetaRadomization(0.02)
    C.setup(0.08)
    C._function()
