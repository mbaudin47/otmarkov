# -*- coding: utf-8 -*-
"""
@author: Michaël Baudin

Réalise une simulation de Monte-Carlo simple sur une chaîne de Markov.

Définit la classe MarkovChain pour qui implémente une chaîne de Markov à
sauts discrets.
"""

import openturns as ot


class MarkovChain:
    """A Markov chain class."""

    def __init__(self, step_function, distribution, number_of_steps, initial_state):
        """
        Create a new Markov Chain.

        Parameters
        ----------
        step_function : function
            The function which performs the step
        distribution : ot.Distribution
            The distribution of the state
        number_of_steps : int
            The number of steps within the chain
        initial_state : float
            The value of the initial state
        """
        self.step_function = step_function
        self.distribution = distribution
        self.number_of_steps = number_of_steps
        self.initial_state = initial_state
        # Dimension of the input random vector of the step function.
        self.input_step_dimension = self.distribution.getDimension()
        # Créée la fonction pour la chaîne
        self.aggregated_dimension = self.input_step_dimension * self.number_of_steps
        # Aggregate the random inputs for all states by repetition.
        list_of_distributions = [self.distribution] * self.number_of_steps
        self.aggregated_distribution = ot.BlockIndependentDistribution(
            list_of_distributions
        )

        def myChainFunction(X):
            X = ot.Point(X)
            Y = self.initial_state
            for i in range(self.number_of_steps):
                index_start = i * self.input_step_dimension
                index_stop = (i + 1) * self.input_step_dimension
                Xn = X[index_start:index_stop]
                Y = self.step_function(Y, Xn)
            return [Y]

        aggregated_dimension = self.aggregated_distribution.getDimension()
        self.function = ot.PythonFunction(aggregated_dimension, 1, myChainFunction)

    def getInputDistribution(self):
        """
        Return the input distribution.

        Returns
        -------
        aggregated_distribution : ot.Distribution
            The distribution of the random state, for all steps.

        """
        return self.aggregated_distribution

    def getFunction(self):
        """
        Return the function for all steps.

        This function takes the initial state as input
        and returns the final state as output.
        It performs a loop over the number of steps.
        At each step, the new state is computed from the curent
        random vector and the old step.

        Returns
        -------
        function: ot.PythonFunction
            The model for all steps.

        """
        return self.function

    def getOutputRandomVector(self):
        """
        Return the output random vector.

        Returns
        -------
        myOutputRV : ot.CompositeRandomVector
            The random vector which the output of the Markov chain.

        """
        myInputRV = ot.RandomVector(self.aggregated_distribution)
        myOutputRV = ot.CompositeRandomVector(self.function, myInputRV)
        return myOutputRV
