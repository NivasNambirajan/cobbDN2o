import numpy as np
import pandas as pd
import sys
import math
from scipy.stats import lognorm
from scipy.special import erfinv
import matplotlib.pyplot as plt


class LinearAdstockTransformation(object):
    """
    This class performs adstock transformation on variables with same adstock transformation parameters
    This class implements a linear adstock build
    """
    def __init__(self, variables, week_to_peak, length, retention_rate):
        """
        Args:
            variables: dataframe of variables with same adstock parameters
            week_to_peak: int
            length: int, length of transformed adstock
            retention_rate: float, decay/carry-over rate
        """

        self.week_to_peak = week_to_peak
        self.length = length
        self.retention_rate = retention_rate
        self.variables = np.matrix(variables)

        if self.variables.shape[0] < self.length:
            sys.exit('Number of weeks is less than adstock length. Program Stops')


    def linear_build_adstock_transform(self):
        """
        This function implements a linear adstock build
        """
        adstock_build = np.arange(1, self.week_to_peak + 1) / self.week_to_peak
        adstock_decay = self.retention_rate ** np.arange(1, self.length - self.week_to_peak + 1)
        adstock_coeff = np.concatenate([adstock_build, adstock_decay])[:, np.newaxis]
        adstock_coeff_normalized = adstock_coeff / sum(adstock_coeff)

        # final_adstock = np.zeros((max(self.length, self.variables.shape[0]), self.variables.shape[1]))
        final_adstock = np.zeros(self.variables.shape)
        for week in range(len(self.variables)):
            adstock = np.squeeze(np.asarray(self.variables[week, :]), axis=0) * adstock_coeff_normalized
            adstock = np.concatenate([np.zeros((week, self.variables.shape[1])), adstock], axis=0)

            if len(adstock) < len(final_adstock):
                final_adstock[:len(adstock), :] += adstock
            else:
                final_adstock += adstock[:len(final_adstock), :]

        return final_adstock

class LogNormAdstockTransformation(object):
    """
    This class performs adstock transformation on variables with same adstock transformation parameters
    This implements a lognorm adstock transformation
    """

    def __init__(self, variables, week_to_peak, length):
        """
        Args:
            variables: dataframe of variables with same adstock parameters
            week_to_peak: int
            length: int, length of transformed adstock
        """

        self.week_to_peak = week_to_peak
        self.length = length
        self.variables = np.matrix(variables)

        if self.variables.shape[0] < self.length:
            sys.exit('Number of weeks is less than adstock length. Program Stops')

    def get_lognorm_mu_sigma(self):
        """
        This function calculates mu and sigma for a lognorm function
        lognorm's mode is peak week
        the x with CDF = 0.95 is length
        """
        mu, sigma = None, None

        a, b, c = 1, 2 ** 0.5 * erfinv(0.9), np.log(self.week_to_peak) - np.log(self.length)
        delta = (b ** 2 - 4 * a * c) ** 0.5

        if delta > 0:
            root_one = (-b + (b ** 2 - 4 * a * c) ** 0.5) / 2
            root_two = (-b - (b ** 2 - 4 * a * c) ** 0.5) / 2
            sigma = max(root_one, root_two)
        elif delta == 0:
            sigma = -b
        else:
            print('Error solving quadratic equation')

        mu = np.log(self.week_to_peak) + sigma ** 2
        return mu, sigma


    def lognorm_build_adstock_transform(self):
        """
        This function implements a lognorm adstock build
        """
        mu, sigma = self.get_lognorm_mu_sigma()
        distribution = lognorm(s=sigma, scale=math.exp(mu))
        weeks = np.arange(1, self.length + 1)
        adstock_coeff_normalized = distribution.pdf(weeks) / sum(distribution.pdf(weeks))
        adstock_coeff_normalized = adstock_coeff_normalized[:, np.newaxis]

        final_adstock = np.zeros(self.variables.shape)
        for week in range(len(self.variables)):
            adstock = np.squeeze(np.asarray(self.variables[week, :])) * adstock_coeff_normalized
            adstock = np.concatenate([np.zeros((week, self.variables.shape[1])), adstock], axis=0)

            if len(adstock) < len(final_adstock):
                final_adstock[:len(adstock), :] += adstock
            else:
                final_adstock += adstock[:len(final_adstock), :]

        return final_adstock


# var = [3300000,	3600000,	3900000,	4200000,	2250000,	1650000,	1800000,	1950000,	700000,	750000,	550000,	600000,	650000,	4620000,	2700000,	2400000,	2100000,	1800000,	750000,	1200000,	1050000,	900000,	250000,	200000,	150000,	100000,	50000]
# var = [3300000]
# var = pd.read_csv(r'./test_stack.csv')
# linear_adstock_transformer = LinearAdstockTransformation(var, 8, 22, 0.6)
# lognorm_adstock_transformer = LogNormAdstockTransformation(var, 8, 22)
# adstock_linear = linear_adstock_transformer.linear_build_adstock_transform()
# adstock_lognorm = lognorm_adstock_transformer.lognorm_build_adstock_transform()
# plt.plot(var, color='y', label='original')
# plt.plot(adstock_linear, color='r', label='linear')
# plt.plot(adstock_lognorm, color='b', label='lognorm')
# plt.legend(loc='upper left')
# plt.show()



