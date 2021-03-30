# -*- coding: utf-8 -*-
"""
@author: Michaël Baudin

Réalise une simulation de Monte-Carlo simple sur une chaîne de Markov.

Définit la classe MarkovChain pour simplifier le traitement de
la chaîne.
"""

import openturns as ot
import numpy as np
import otmarkov


class MarkovChainRandomVector(ot.PythonRandomVector):
    """Create a Markov chain random vector."""

    def __init__(self, step_function, distribution, number_of_steps, initial_state):
        """
        Create a Markov chain random vector.

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
        self.markov_chain = otmarkov.MarkovChain(step_function, distribution,
                                                 number_of_steps, initial_state)
        self.randomvector = self.markov_chain.getOutputRandomVector()
        aggregated_distribution = self.markov_chain.getInputDistribution()
        aggregated_dimension = aggregated_distribution.getDimension()
        super(MarkovChainRandomVector, self).__init__(aggregated_dimension)
        return None

    def getRealization(self):
        """
        Generate a random realization of the chain.

        Returns
        -------
        realization: ot.Point(d)
            The output of the Markov chain after all steps.

        """
        return self.randomvector.getRealization()

    def getSample(self, size):
        """
        Generate a sample of the Markov chain.

        Parameters
        ----------
        size : int
            The size of the sample.

        Returns
        -------
        sample: ot.Sample(size, d)
            The sample from the Markov chain.

        """
        return self.randomvector.getSample(size)
