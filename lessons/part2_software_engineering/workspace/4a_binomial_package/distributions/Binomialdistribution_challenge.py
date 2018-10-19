import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from .Generaldistribution import Distribution

DATAFILE_PATH = 'numbers_binomial.txt'


class Binomial(Distribution):

    """ Binomial distribution class for calculating and
    visualizing a Binomial distribution.

    Attributes:
        mean (float) representing the mean value of the distribution
        stdev (float) representing the standard deviation of the distribution
        data_list (list of floats) a list of floats to be extracted from the data file
        p (float) representing the probability of an event occurring

    """

    def __init__(self, p, n):
        """
        A binomial distribution is defined by two variables:
            the probability of getting a positive outcome
            the number of trials

        If you know these two values, you can calculate the mean and the standard deviation

        For example, if you flip a fair coin 25 times, p = 0.5 and n = 25
        You can then calculate the mean and standard deviation with the
        following formula:
            mean = p * n
            standard deviation = sqrt(n * p * (1 - p))
        """
        Distribution.__init__(self)
        self.p = p
        self.n = n
        self.mean = self.calculate_mean()
        self.stdev = self.calculate_stdev()

    def calculate_mean(self):
        """Function to calculate the mean from p and n

        Args:
            None

        Returns:
            float: mean of the data set

        """
        return self.p * self.n

    def calculate_stdev(self):
        """Function to calculate the standard deviation from p and n.

        Args:
            None

        Returns:
            float: standard deviation of the data set

        """
        return math.sqrt(self.n * self.p * (1 - self.p))

    def replace_stats_with_data(self):
        """
        Function to calculate p and n from the data set. The function updates
        the p and n variables of the object.

        Args:
            None

        Returns:
            float: the p value
            float: the n value

        """
        self.read_data_file(DATAFILE_PATH)

        self.n = len(self.data)
        self.p = np.mean(self.data)

        return self.p, self.n

    def plot_bar(self):
        """
        Function to output a histogram of the instance variable data using
        matplotlib pyplot library.

        Args:
            None

        Returns:
            None
        """
        plt.hist(self.data)

    #TODO: Calculate the probability density function of the binomial distribution
    def pdf(self, k):
        """Probability density function calculator for the binomial distribution.

        Args:
            k (float): point for calculating the probability density function


        Returns:
            float: probability density function output
        """
        return binom.pmf(k, self.n, self.p)

    def plot_pdf(self):
        """Function to plot the pdf of the binomial distribution

        Args:
            None

        Returns:
            list: x values for the pdf plot
            list: y values for the pdf plot

        """
        x = np.arange(0, self.n + 1)
        y = self.pdf(x)

        plt.bar(x, y)
        plt.title('Mass probability function plot.')
        plt.xlabel('k')
        plt.ylabel('P(k)')

        return x.tolist(), y.tolist()

    # write a method to output the sum of two binomial distributions. Assume both distributions have the same p value.
    def __add__(self, other):
        """Function to add together two Binomial distributions with equal p

        Args:
            other (Binomial): Binomial instance

        Returns:
            Binomial: Binomial distribution

        """
        try:
            assert self.p == other.p, 'p values are not equal'
        except AssertionError as error:
            raise
        return Binomial(self.p, self.n + other.n)

    def __repr__(self):
        """Function to output the characteristics of the Binomial instance

        Args:
            None

        Returns:
            string: characteristics of the Binomial object

        """
        return 'mean {:.2}, standard deviation {:.2}, p {:.2}, n {}'.format(self.mean,
                                                                  self.stdev,
                                                                  self.p,
                                                                  self.n)
