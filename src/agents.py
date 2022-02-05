import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import bernoulli, beta

random.seed(100)
np.random.seed(100)


def argmax(vec):
    """Chose randomly one index where the input realizes it's maximum."""
    return np.random.choice((vec == vec.max()).nonzero()[0])


class Agent:
    """Base properties of all multi-armed bandits."""

    def __init__(self, n_arms):
        """
        n_arms: int
            Number of arms
        """
        self.n_arms = n_arms
        self.tries = np.zeros(n_arms)
        self.successes = np.zeros(n_arms)

    @property
    def _total_tries(self):
        return self.tries.sum()

    def add_observation(self, arm, reward):
        """An observation consists of a pair `(arm, reward)`.
        arm: int
            Chosen arm of the observation
        reward: float
            Reward of the observation
        """
        self.tries[arm] += 1
        self.successes[arm] += reward

    def select_arm(self):
        """Placeholder way of selecting an arm."""
        arm = self._total_tries % self.n_arms
        return arm


class GreedyExploreFirst(Agent):
    """We explore for `learning_stop` steps then exploit the reward that gave the most expected reward so far."""

    def __init__(self, n_arms, learning_stop):
        """
        n_arms: int
            Number of arms
        learning_stop: int
            Time step until we decide to stop chosing arms randomly
        """
        super().__init__(n_arms)
        self.learning_stop = learning_stop

    def select_arm(self):
        # if learning hasn't stopped yet, chose random arm
        if self._total_tries < self.learning_stop:
            arm = (
                self._total_tries % self.n_arms
            )  # can also replace with arm = random.randint(0, self.n_arms - 1)
        # else chose arm with most success rate
        else:
            success_rate = self.successes / self.tries
            arm = argmax(success_rate)
        arm = int(arm)
        return arm


class EpsilonGreedy(Agent):
    """At each step, we throw an $\epsilon$-weighted coin toss do decide wether to explore or exploit."""

    @property
    def _epsilon(self):
        # value taken from https://arxiv.org/pdf/1904.07272.pdf theorem 1.6
        epsilon = (self.n_arms * np.log(self._total_tries + 1) / self._total_tries) ** (
            1 / 3
        )
        epsilon = min(epsilon, 1)
        return epsilon

    def select_arm(self):
        # if epsilon-weighted coin toss is true, explore

        if self._total_tries < self.n_arms:
            arm = self._total_tries
        else:
            epsilon = self._epsilon
            if bernoulli(epsilon).rvs():
                arm = random.randint(0, self.n_arms - 1)
            # else exploit (chose arm with most success rate)
            else:
                success_rate = self.successes / (self.tries + 1e-6)
                arm = argmax(success_rate)
        arm = int(arm)
        return arm


class UCB1(Agent):
    """This uses the concept of 'optimism under uncertainty'.

    We assume the expected reward of each arm is the upper confidence bound (UCB) of the arm so far.

    The confidence radius is calculated using Hoeffding’s Inequality."""

    @property
    def _confidence_radius(self):
        radius = np.sqrt(2 * np.log(self._total_tries) / self.tries)
        return radius

    @property
    def _UCB(self):
        success_rate = self.successes / self.tries
        return success_rate + self._confidence_radius

    def select_arm(self):
        # Start playing each arm once
        if self._total_tries < self.n_arms:
            arm = self._total_tries
        else:
            UCB = self._UCB
            arm = argmax(UCB)
        arm = int(arm)
        return arm


class BayesUCB(Agent):
    """Assess UCB using knowledge of the Bernoulli distribution and it's conjugate prior, the Beta distribution."""

    def __init__(self, n_arms, c=2):
        """
        n_arms: int
            Number of arms
        c: float
            Amount of standard deviations considered for the UCB
        """
        super().__init__(n_arms)
        self.c = c  # Amount of standard deviations considered for UCB
        self._params_a = np.ones(n_arms)  # Params a of Beta
        self._params_b = np.ones(n_arms)  # Params b of Beta

    def add_observation(self, arm, reward):
        """
        arm: int
            Arm chosen in the observation
        reward: bool
            Reward obtained in the observation
        """
        super().add_observation(arm, reward)
        self._params_a[arm] += reward
        self._params_b[arm] += 1 - reward

    def select_arm(self):
        # Start playing each arm once
        if self._total_tries < self.n_arms:
            arm = self._total_tries
        else:
            bayes_UCB = self._params_a / (self._params_a + self._params_b)  # Mean
            bayes_UCB += self.c * beta.std(
                self._params_a, self._params_b
            )  # Confidence Radius
            arm = argmax(bayes_UCB)
        arm = int(arm)
        return arm


class ThompsonSampling(Agent):
    """Every arm is supposed to be a Beta distribution with parameters `a` = successes, `b` = failures.

    At each step we sample from the Beta distributions of all arms and chose an arm with the largest sample."""

    def __init__(self, n_arms):
        """
        n_arms: int
            Number of arms
        """
        super().__init__(n_arms)
        self._params_a = np.ones(n_arms)  # Params a of Beta
        self._params_b = np.ones(n_arms)  # Params b of Beta

    def add_observation(self, arm, reward):
        """
        arm: int
            Arm chosen in the observation
        reward: bool
            Reward obtained in the observation
        """
        super().add_observation(arm, reward)
        self._params_a[arm] += reward
        self._params_b[arm] += 1 - reward

    def select_arm(self):
        # Start playing each arm once
        if self._total_tries < self.n_arms:
            arm = self._total_tries
        else:
            # Sample from the beta distribution and chose the argmax
            vec = beta(self._params_a, self._params_b).rvs()
            arm = np.random.choice((vec == vec.max()).nonzero()[0])
        arm = int(arm)
        return arm
