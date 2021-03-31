# -*- coding: utf-8 -*-
# Copyright 2018 - 2019 EDF.
"""
Test de la classe MarkovChain.
"""

import openturns as ot
import unittest
from numpy.testing import assert_allclose
import otmarkov
import numpy as np


def model(X):
    """
    The function which performs the step.
    Parameters
    ----------
    X : ot.Point(4)
        The input of the model.

    Returns
    -------
    new_state : ot.Point(1)
        The new state.
    """
    P = X[0]
    Q = X[1]
    R = X[2]
    state = X[3]
    new_state = state + P * Q + R
    return [new_state]


class TestMarkovChain(unittest.TestCase):
    def test_PQR(self):

        model_py = ot.PythonFunction(4, 1, model)

        # Create a parametric function from the model
        # The input is random, the parameter is the state, the output is the new state.
        initial_state = ot.Point([0.0])
        indices = [3]
        step_function = ot.ParametricFunction(model_py, indices, initial_state)

        # Evaluate step function
        P = 1.794
        Q = 2.387
        R = -2.123
        X = [P, Q, R]
        Y = step_function(X)
        np.testing.assert_allclose(Y, [2.15928], rtol=1.0e-4)

        # Create the random vector.
        P = ot.Normal()
        Q = ot.Normal()
        R = ot.WeibullMin()
        distribution = ot.ComposedDistribution([P, Q, R])

        # Create the Markov chain
        number_of_steps = 4
        markov_chain = otmarkov.MarkovChain(
            step_function, distribution, number_of_steps, initial_state
        )

        random_vector = markov_chain.getCompositeRandomVector()

        # Estimate the mean with Monte-Carlo
        sampleSize = 10000
        sample = random_vector.getSample(sampleSize)
        sample_mean = sample.computeMean()[0]
        print("sample_mean=", sample_mean)
        relativeError = 10.0 / np.sqrt(sampleSize)
        mu_exact = 4.0
        assert_allclose(sample_mean, mu_exact, relativeError)

    def test_PQR_simulation(self):
        model_py = ot.PythonFunction(4, 1, model)

        # Create a parametric function from the model
        # The input is random, the parameter is the state, the output is the new state.
        initial_state = ot.Point([0.0])
        indices = [3]
        step_function = ot.ParametricFunction(model_py, indices, initial_state)

        # Create the random vector.
        P = ot.Normal()
        Q = ot.Normal()
        R = ot.WeibullMin()
        distribution = ot.ComposedDistribution([P, Q, R])

        # Create the Markov chain
        number_of_steps = 4
        markov_chain = otmarkov.MarkovChain(
            step_function, distribution, number_of_steps, initial_state
        )
        result = markov_chain.simulate()
        initial_state = result.getInitialState()
        print("initial_state=", initial_state)
        final_state = result.getFinalState()
        print("final_state=", final_state)
        history = result.getHistory()
        print("history=")
        for point in history:
            print(point)
        number_of_result_steps = result.getNumberOfSteps()
        print("number_of_steps=", number_of_result_steps)
        assert number_of_result_steps == number_of_steps


if __name__ == "__main__":
    unittest.main()
