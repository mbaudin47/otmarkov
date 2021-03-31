# -*- coding: utf-8 -*-
"""
@author: Michaël Baudin

Defines a Piecewise Deterministic Markov Process on finite horizon.
"""

import openturns as ot
import otmarkov


class MarkovProcess:
    """A Markov process class."""

    def __init__(
        self,
        step_function,
        distribution,
        stop_callback,
        maximum_number_of_steps,
        initial_state,
    ):
        """
        Create a new Markov process.

        Parameters
        ----------
        step_function : function
            The function which performs the step
        distribution : ot.Distribution
            The distribution of the state
        maximum_number_of_steps : int
            The maximum number of steps in the process.
        stop_callback : function
            The function which evaluates the stoping rule.
        initial_state : float
            The value of the initial state
        """
        self.step_function = step_function
        self.distribution = distribution
        self.stop_callback = stop_callback
        self.maximum_number_of_steps = maximum_number_of_steps
        self.initial_state = initial_state
        return None

    def simulate(self):
        """
        Return a realization of the process.

        Returns
        -------
        state: ot.Point
            The result of the process.

        """
        state = self.initial_state
        history = [state]
        for i in range(self.maximum_number_of_steps):
            X = self.distribution.getRealization()
            self.step_function.setParameter(state)
            state = self.step_function(X)
            history.append(state)
            # Shall we stop?
            must_stop = self.stop_callback(state)
            if must_stop:
                break
        result = otmarkov.MarkovChainResult(history)
        return result
