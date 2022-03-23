"""
Defines interfaces for generative models.
"""
from abc import ABC, abstractmethod


class GenerativeModel(ABC):

    @abstractmethod
    def generate_discrete_trajectories(self, n_steps):
        """
        Produces a discrete trajectory
        :return:
        """

        raise NotImplementedError('Overload this with something that returns discrete trajectories.')

    @abstractmethod
    def assign_structure(self, state_index):
        """
        Given a state index, return a corresponding structure.

        :parameter state_index: State index to return a structure for

        :return: Structure
        """

        raise NotImplementedError('Overload this with something that returns a discrete trajectory.')

    @abstractmethod
    def backmap(self, _dtrajs):
        """
        Given discrete trajectories, backmap them to full coordinate representations.

        :param _dtrajs: Array-like, Discrete trajectories
        :return:
        """