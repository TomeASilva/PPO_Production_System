from collections import deque
import numpy as np

class EpBuffer:
    """
    Class that stores the state transition information of an episode
    """

    def __init__(self):
        self.memory = deque()
    
    def add_transition(self, transition):
        """
        Arguments:
        transition -> Tuple (s, a, s', reward) (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.float64 )
        """
        self.memory.append(transition)

    @staticmethod
    def compute_Qsa(rewards, gamma):  # O(n^2)
        """
        Computes the sample value function (ground truth) for every single state action pair of an episode

        Arguments:
        rewards -> object that contain all the rewards from the episode from t = 0 to t = len(rewards)
        gamma -> float, discount factor for the rewards

        Returns:
        Qsa -> List

        """
        Qsa = []
        for i in range(len(rewards)):
            partial_Qsa = 0
            t = 0
            for j in range(i, len(rewards)):

                partial_Qsa += rewards[j] * (gamma ** t)
                t += 1

            Qsa.append(partial_Qsa)

        return Qsa
    
    def unroll_memory(self, gamma):
        """
        Unrolls the states transitions information so that states , actions, next_states, rewards and Qsa's
        are separeted into different numpy arrays

        Returns:
        states -> numpy array (state dimension, num of state transitions)
        actions -> numpy array (action dimension, num of state transitions)
        next_states -> numpy array (state dimension, num of state transitions)
        rewards -> numpy array (num of state transitions, )
        qsa -> numpy array (num of state transitions, )
        """

        states, actions, next_states, rewards = zip(*self.memory)
        qsa = self.compute_Qsa(rewards, gamma)
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32).reshape(-1, 1)
        next_states = np.asarray(next_states)
        rewards = np.asarray(rewards, dtype=np.float32)
        qsa = np.asarray(qsa, dtype=np.float32).reshape(-1, 1)

        # print(f"States: {states.shape}")
        # print(f"actions: {actions.shape}")
        # print(f"next_states: {next_states.shape}")
        # print(f"rewards: {rewards.shape}")
        # print(f"qsa: {qsa.shape}")
        self.memory = deque()
        return states, actions, next_states, rewards, qsa
