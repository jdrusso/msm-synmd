import numpy as np
import tqdm.auto as tqdm
from .models import GenerativeModel


class GenerativeMarkovModel(GenerativeModel):
    """
    Class for generating discrete trajectories from a transition matrix, and backmapping them to full-coordinate
    synthetic MD trajectories.

    TODO:
        - Log dT somewhere/somehow
    """

    def __init__(self, transition_matrix, seed=None):

        self.structure_map = None

        self.transition_matrix = transition_matrix
        self.rng = np.random.default_rng(seed=seed)

    def generate_discrete_trajectories(self, n_steps, initial_states, n_trajectories=1):
        """
        Generate discrete trajectories.

        :param n_steps:
        :param initial_states:
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
        with tqdm.tqdm(total=n_trajectories * (n_steps-2), desc="Generating trajectories...") as pbar:

            # for _trajectory in tqdm.trange(n_trajectories, desc="Trajectory"):
            for _trajectory in range(n_trajectories):
                # for _step in tqdm.trange(1, n_steps, desc="Step", leave=False, disable=True):
                for _step in range(1, n_steps):

                    previous_state = generated_trajectories[_trajectory, _step-1]
                    transition_probabilities = self.transition_matrix[previous_state]

                    next_state = self.rng.choice(states, p=transition_probabilities)
                    generated_trajectories[_trajectory, _step] = next_state

                    pbar.update(1)

        return generated_trajectories

    def backmap(self, _dtrajs, subsample=1):
        """
        Map discrete trajectories to full-coordinate trajectories.

        :param _dtrajs: Array-like of discrete integer trajectories
        :param subsample: Only backmap every Nth frame
        :return: Full-coordinate trajectories
        """

        _continuous_trajectories = []

        for dtraj in tqdm.tqdm(_dtrajs, desc="Backmapping trajectories.."):

            _continuous_trajectory = map(self.assign_structure,
                                         # tqdm.tqdm(dtraj, desc="\t Backmapping", leave=False))
                                         dtraj[::subsample])

            _continuous_trajectories.append(list(_continuous_trajectory))

        return np.array(_continuous_trajectories)

    def write_trajectory(self):
        raise NotImplementedError()

    def assign_structure(self, state_index):
        raise NotImplementedError()
