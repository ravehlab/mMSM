"""Base class for all Trajectory Samplers"""

# Author: Kessem Clein <kessem.clein@mail.huji.ac.il>

from abc import ABC, abstractmethod

class BaseTrajectorySampler(ABC):
    @property
    @abstractmethod
    def timestep_size(self):
        pass

    @abstractmethod
    def sample_from_states(self, states, sample_len, n_samples, sample_interval=1):
        """sample_from_states.
        Get a set of discrete trajectories (sequences of states), originating from a given set of
        states.


        Parameters
        ----------
        states : Iterable[int]
            An iterable of the ids of the states to start samples from.
        sample_len : int
            The length of each sample trajectory.
        n_samples : int
            The number of trajectories to sample from each state in states.
        sample_interval : int
            The interval between each state taken in the returned trajectories. 
            For example, if sample_interval==2, the trajectory returned from the actual trajectory
            (1,2,3,4,5,6,7) would be (1,3,5,7).
            If sample_interval is 1 (default) the full trajectories are returned.

        Returns:
        --------
        dtrajs : list
            A list containing len(states)*n_samples lists of length sample_len. Each list is a
            discrete trajectory (a sequence of states).
        """
        pass

    @abstractmethod
    def get_initial_sample(self, sample_len, n_samples, sample_interval=1):
        """get_initial_sample
        Get an initial set of discrete trajectories (sequences of states).

        Parameters
        ----------
        sample_len : int
            The length of each sample trajectory.
        n_samples : int
            The number of trajectories to sample.
        sample_interval : int
            The interval between each state taken in the returned trajectories. 
            For example, if sample_interval==2, the trajectory returned from the actual trajectory
            (1,2,3,4,5,6,7) would be (1,3,5,7).
            If sample_interval is 1 (default) the full trajectories are returned.

        Returns:
        --------
        dtrajs : list
            A list containing len(states)*n_samples lists of length sample_len. Each list is a
            discrete trajectory (a sequence of states).
        """
        pass
