import math
import numpy as np

class Normalizer:
    def __init__(self, stretch_type, scale_factor, min_value, max_value):
        self.stretch_type = stretch_type
        self.scale_factor = scale_factor
        self.min_value = min_value
        self.max_value = max_value

    def stretch(self, data):
        MAX = self.max_value
        MIN = self.min_value
        data[data<MIN] = MIN
        data[data>MAX] = MAX
        if self.stretch_type == 'log':
            return np.log10(self.scale_factor * ((data - self.min_value) / \
                                 (self.max_value - self.min_value)) + 1) / \
                                 math.log10(self.scale_factor)
        elif self.stretch_type == 'asinh':
            return np.arcsinh(self.scale_factor * data) / \
                              math.asinh(self.scale_factor * self.max_value)
        elif self.stretch_type == 'pow':
            return np.power((data - self.min_value) / \
                            (self.max_value - self.min_value),
                            1 / float(self.scale_factor))
        elif self.stretch_type == 'linear':
            return (data - self.min_value) / (self.max_value - \
                                                   self.min_value)
        elif self.stretch_type == 'sigmoid':
            return (1 / (1 + np.exp(-self.scale_factor * \
                                np.sqrt((data - self.min_value) /\
                                (self.max_value - self.min_value)))) \
                                - 1 / 2) * 2
        else:
            raise ValueError('Unknown stretch_type : %s' % self.stretch_type)

    def unstretch(self, data):
        if self.stretch_type == 'log':
            return self.min_value + (self.max_value - self.min_value) * \
                   (np.power(data * math.log10(self.scale_factor), 10) - 1) / \
                   self.scale_factor
        elif self.stretch_type == 'asinh':
            return np.sinh(data * math.asinh(self.scale_factor * \
                   self.max_value)) / self.scale_factor
        elif self.stretch_type == 'pow':
            return np.power(data, self.scale_factor) * \
                   (self.max_value - self.min_value) + self.min_value
        elif self.stretch_type == 'linear':
            return data * (self.max_value - self.min_value) + self.min_value
        elif self.stretch_type == 'sigmoid':
            return np.square(np.log(-1 + 2 / (data + 1)) / self.scale_factor) *\
                   (self.max_value - self.min_value) + self.min_value
        else:
            raise ValueError('Unknown stretch_type : %s' % self.stretch_type)