import numpy as np



class BetaRadomization:

    def __init__(self, beta, seed=None, param_set='DENSE'):

        if seed is not None:
            # fix random seed
            np.random.seed(seed)

        self.noise_mean = 0.0
        self.noise_std = 0.0

        self.beta = beta

        if param_set == 'DENSE_use_n_heights':

            magnitude = 0.05

            mhf = 2  # max horzontal frequency
            mvf = 5  # max vertical frequency

            # sample number of fourier components, for angle and height independendently
            n_angles = np.random.randint(6, 10)
            n_heights = np.random.randint(3, 5)

            # sample frequencies
            self.frequencies_angle = np.random.randint(1, mhf, size=n_angles)
            self.frequencies_height = np.random.randint(0, mvf, size=n_heights)

            # sample random offsets to one another
            self.offset_angle = np.random.uniform(0, 2 * np.pi, size=n_angles)
            self.offset_height = np.random.uniform(0, 2 * np.pi, size=n_heights)

            # sample intensities
            self.intensity_angle = np.random.uniform(0, magnitude / n_angles, size=n_angles)
            self.intensity_height = np.random.uniform(0, magnitude / n_heights, size=n_heights)

        elif param_set == 'DENSE_no_noise':

            magnitude = 0

            mhf = 2  # max horzontal frequency
            mvf = 5  # max vertical frequency

            # sample number of fourier components
            n_components = np.random.randint(6, 10)

            # sample frequencies
            self.frequencies_angle = np.random.randint(1, mhf, size=n_components)
            self.frequencies_height = np.random.randint(0, mvf, size=n_components)

            # sample random offsets to one another
            self.offset_angle = np.random.uniform(0, 2 * np.pi, size=n_components)
            self.offset_height = np.random.uniform(0, 2 * np.pi, size=n_components)

            # sample intensities
            self.intensity_angle = np.random.uniform(0, magnitude / n_components, size=n_components)
            self.intensity_height = np.random.uniform(0, magnitude / n_components, size=n_components)

        elif param_set == 'CVL':

            magnitude = 0.01
            # magnitude = self.beta / 2

            mhf = 2  # max horzontal frequency
            mvf = 5  # max vertical frequency

            # sample number of fourier components
            n_components = np.random.randint(6, 10)

            # sample frequencies
            self.frequencies_angle = np.random.randint(1, mhf, size=n_components)
            self.frequencies_height = np.random.randint(0, mvf, size=n_components)

            # sample random offsets to one another
            self.offset_angle = np.random.uniform(0, 2 * np.pi, size=n_components)
            self.offset_height = np.random.uniform(0, 2 * np.pi, size=n_components)

            # sample intensities
            self.intensity_angle = np.random.uniform(0, magnitude / n_components, size=n_components)
            self.intensity_height = np.random.uniform(0, magnitude / n_components, size=n_components)

        else: # assume param_set == 'DENSE'

            magnitude = 0.05

            mhf = 2 # max horzontal frequency
            mvf = 5 # max vertical frequency

            # sample number of fourier components
            n_components = np.random.randint(6, 10)

            # sample frequencies
            self.frequencies_angle = np.random.randint(1, mhf, size=n_components)
            self.frequencies_height = np.random.randint(0, mvf, size=n_components)

            # sample random offsets to one another
            self.offset_angle = np.random.uniform(0, 2 * np.pi, size=n_components)
            self.offset_height = np.random.uniform(0, 2 * np.pi, size=n_components)

            # sample intensities
            self.intensity_angle = np.random.uniform(0, magnitude / n_components, size=n_components)
            self.intensity_height = np.random.uniform(0, magnitude / n_components, size=n_components)



    def propagate_in_time(self, timestep):

        self.offset_angle += self.frequencies_angle * timestep / 10
        self.offset_height += self.frequencies_height * timestep / 10


    def _function(self, angle, height):

        a, h = angle, height

        output = np.zeros(np.shape(a))

        for fa, fh, oa, oh, ih, ia in zip(self.frequencies_angle,
                                          self.frequencies_height,
                                          self.offset_angle,
                                          self.offset_height,
                                          self.intensity_height,
                                          self.intensity_angle):

            noise = np.abs((ia*np.sin(fa*a+oa)/fa+ih*np.sin(fa*a+fh*h+oh)))

            self.noise_mean = noise.mean()
            self.noise_std = noise.std()

            output += noise

        output += self.beta

        return output


    def get_beta(self, forward, right, height):

        forward = np.where(forward == 0, 0.0001, forward)   # prevent division by zero in next line
        angle = np.tan(np.divide(right, forward))
        ranomized_beta = self._function(angle, height)

        return ranomized_beta