import numpy as np
import tqdm.auto as tqdm
from .models import GenerativeModel


class GenerativeMarkovModel(GenerativeModel):

    def __init__(self, transition_matrix, seed=None):

        self.structure_map = None

        self.transition_matrix = transition_matrix
        self.rng = np.random.default_rng(seed=seed)

    def generate_discrete_trajectories(self, n_steps, initial_states, n_trajectories=1):
        """
        Generate discrete trajectories.

        :param n_steps:
        :param initial_positions:
        :param n_trajectories:
        :return:
        """

        assert len(initial_states) == n_trajectories, \
            "Number of initial positions doesn't match number of trajectories."

        initial_states = np.array(initial_states)
        states = np.arange(self.transition_matrix.shape[0])

        generated_trajectories = np.empty(shape=(n_trajectories, n_steps), dtype=int)
        generated_trajectories[:, 0] = initial_states

        # Generate the discrete trajectory by sampling a Markov chain
        for _trajectory in tqdm.trange(n_trajectories, desc="Trajectory"):
            for _step in tqdm.trange(1, n_steps, desc="Step"):

                previous_state = generated_trajectories[_trajectory, _step-1]
                transition_probabilities = self.transition_matrix[previous_state]

                next_state = self.rng.choice(states, p=transition_probabilities)
                generated_trajectories[_trajectory, _step] = next_state

        return generated_trajectories

    def backmap(self, _dtrajs):
        """
        Map discrete trajectories to full-coordinate trajectories.

        :param _dtrajs: Array-like of discrete integer trajectories
        :return: Full-coordinate trajectories
        """

        _continuous_trajectories = []

        for dtraj in tqdm.tqdm(_dtrajs, desc="Backmapping trajectories.."):
            _continuous_trajectory = map(self.assign_structure,
                                         tqdm.tqdm(dtraj,
                                                   desc="\t Backmapping", leave=False))

            _continuous_trajectories.append(list(_continuous_trajectory))

        return np.array(_continuous_trajectories)

    def assign_structure(self, state_index):
        raise NotImplementedError()
