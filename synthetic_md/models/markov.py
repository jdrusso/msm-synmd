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


class GenerativeHistoryAugmentedMarkovModel(GenerativeMarkovModel):

    """
    Class for generating discrete trajectories from a colored transition matrix, and backmapping them to full-coordinate
    synthetic MD trajectories.

    This is defined for N macrostates, using N+1 transition matrices.
    The extra transition matrix is used for trajectories that cannot be traced back to either state.

    At each step, the transition matrix is selected based on the last-in state.
    """

    def __init__(self, transition_matrices, macrostate_defs, seed=None):
        """

        :param transition_matrices: Transition matrices (1 for uncolored, then 1 for each macrostate)
        :param macrostate_defs: Definition of microstates in each macrostate
        :param seed: Seed for RNG
        """

        self.structure_map = None

        self.rng = np.random.default_rng(seed=seed)

        assert len(transition_matrices) == len(macrostate_defs)+1, \
            "Number of colored transition matrices doesn't match number of macrostates!"

        self.transition_matrices = np.array(transition_matrices)
        self.macrostate_defs = np.array(macrostate_defs)

        # The number of states in each color
        self.color_states = [np.arange(len(x)) for x in self.transition_matrices]

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

        # For an haMSM, this tracks both state index and color at every time
        generated_trajectories = np.empty(shape=(n_trajectories, n_steps, 2), dtype=int)
        generated_trajectories[:, 0] = initial_states

        # Generate the discrete trajectory by sampling a Markov chain
        with tqdm.tqdm(total=n_trajectories * (n_steps - 2), desc="Generating trajectories...") as pbar:

            for _trajectory in range(n_trajectories):

                for _step in range(1, n_steps):

                    previous_state, previous_color = generated_trajectories[_trajectory, _step - 1]

                    transition_probabilities = self.transition_matrices[previous_color][previous_state]

                    try:
                        next_state = self.rng.choice(self.color_states[previous_color], p=transition_probabilities)
                    except ValueError as e:
                        print(f"Error when getting probabilities for "
                              f"last state {previous_state} last color {previous_color}")
                        print(f"Previous 5 steps were {generated_trajectories[_trajectory, _step -5:_step]}")
                        raise e

                    # To determine the next color, check which macrostate this next state is in
                    next_in_macrostate = np.array([np.isin(next_state, x) for x in self.macrostate_defs])
                    if next_in_macrostate.any():
                        # Add +1 because state 0 is last-in none
                        next_color = np.argwhere(next_in_macrostate).flatten()[0] + 1
                    else:
                        # If the next step is NOT in any particular macrostate, just assign it the previous color
                        next_color = previous_color

                    # I know the color at this time, and I'm about to make a step
                    # That next step will be in my current color's definition

                    # 1. Take the previous state and color
                    # 2. Using the previous color, pick my haMSM
                    # 3. Using the previous state, pick my row
                    #       If I switched MSMs (i.e. a new color), how do I know which state to go to in the new haMSM?
                    #       Maybe I DO need consistent state definitions between the two, then I'm just adjusting
                    #       the transition probabilities when I switch colors.
                    #       NOTE: For now, assume consistent state definitions
                    # 4. Choose my next state
                    # 5. Determine my next color
                    # 6. Next state and color

                    generated_trajectories[_trajectory, _step] = [next_state, next_color]

                    pbar.update(1)

        return generated_trajectories

