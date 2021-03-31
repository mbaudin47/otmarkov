# -*- coding: utf-8 -*-
"""
A class to define a Markov chain result.
"""


class MarkovChainResult:
    """The result of a Markov chain simulation."""

    def __init__(self, history):
        """
        Create the result of a MarkovChain simulation.

        Parameters
        ----------
        history : list of ot.Point(d)
            The sequence of states in the simulation.

        Returns
        -------
        None.

        """
        self.history = history

    def getInitialState(self):
        """
        Return the initial state of the chain.

        Returns
        -------
        state : ot.Point(d)
            The initial state.

        """
        return self.history[0]

    def getFinalState(self):
        """
        Return the final state of the chain.

        Returns
        -------
        state : ot.Point(d)
            The final state.

        """
        return self.history[-1]

    def getNumberOfSteps(self):
        """
        Return the number of steps in the chain.

        This is the length of the history - 1.

        Returns
        -------
        number_of_steps : int
            The number of steps.

        """
        number_of_steps = len(self.history) - 1
        return number_of_steps

    def getHistory(self):
        """
        Return the sequence of states in the chain.

        Returns
        -------
        history : list of ot.Point(d)
            The sequence of states.

        """
        return self.history

