# -*- coding: utf-8 -*-
"""
@author: Michaël Baudin

Defines a Markov chain class with a given number of transitions.
"""

import openturns as ot
import otmarkov


class MarkovChain:
    """A Markov chain class."""

    def __init__(self, step_function, distribution, number_of_steps, initial_state):
        """
        Create a new Markov chain.

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
        # Check dimension of the state
        parameter_dimension = step_function.getParameterDimension()
        state_dimension = initial_state.getDimension()
        if parameter_dimension != state_dimension:
            raise ValueError("The parameter dimension of the step function is %d"
                             "but the dimension of the state is %d" % (
                                parameter_dimension, state_dimension))
        #
        self.step_function = step_function
        self.distribution = distribution
        self.number_of_steps = number_of_steps
        self.initial_state = initial_state
        return None

    def getCompositeRandomVector(self):
        """
        Return the output random vector.

        Returns
        -------
        myOutputRV : ot.CompositeRandomVector
            The random vector which the output of the Markov chain.

        """
        # Dimension of the input random vector of the step function.
        input_step_dimension = self.distribution.getDimension()
        # Créée la fonction pour la chaîne
        aggregated_dimension = input_step_dimension * self.number_of_steps
        # Aggregate the random inputs for all states by repetition.
        list_of_distributions = [self.distribution] * self.number_of_steps
        aggregated_distribution = ot.BlockIndependentDistribution(
            list_of_distributions
        )

        def myChainFunction(X):
            X = ot.Point(X)
            state = self.initial_state
            for i in range(self.number_of_steps):
                # Get the random input for this step
                index_start = i * input_step_dimension
                index_stop = (i + 1) * input_step_dimension
                Xn = X[index_start:index_stop]
                # Update the state
                self.step_function.setParameter(state)
                # Compute and update the state
                state = self.step_function(Xn)
            return state

        aggregated_dimension = aggregated_distribution.getDimension()
        function = ot.PythonFunction(aggregated_dimension, 1, myChainFunction)
        output_description = self.step_function.getOutputDescription()
        function.setOutputDescription(output_description)
        myInputRV = ot.RandomVector(aggregated_distribution)
        composite_random_vector = ot.CompositeRandomVector(function, myInputRV)
        return composite_random_vector

    def getStateDimension(self):
        """
        Return the input distribution.

        Returns
        -------
        aggregated_distribution : ot.Distribution
            The distribution of the random state, for all steps.

        """
        return self.initial_state.getDimension()

    def simulate(self):
        """
        Simulate a trajectory.

        Returns
        -------
        result : ot.MarkovChainResult
            The result of the simulation.
        """
        state = self.initial_state
        history = [state]
        for i in range(self.number_of_steps):
            Xn = self.distribution.getRealization()
            self.step_function.setParameter(state)
            state = self.step_function(Xn)
            history.append(state)
        result = otmarkov.MarkovChainResult(history)
        return result
