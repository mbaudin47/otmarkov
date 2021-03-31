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


class TestMarkovChain(unittest.TestCase):
    def test_PQR(self):
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

        model_py = ot.PythonFunction(4, 1, model)

        # Create a parametric function from the model
        # The input is random, the parameter is the state, the output is the new state.
        initial_state = [0.0]
        indices = [3]
        step_function = ot.ParametricFunction(model_py, indices, initial_state)

        # Evaluate step function
        P = 1.794
        Q = 2.387
        R = -2.123
        X = [P, Q, R]
        Y = step_function(X)
        np.testing.assert_allclose(Y, [2.15928], rtol=1.e-4)

        # Create the random vector.
        P = ot.Normal()
        Q = ot.Normal()
        R = ot.WeibullMin()
        distribution = ot.ComposedDistribution([P, Q, R])

        # Create the Markov chain
        number_of_steps = 4
        myMCF = otmarkov.MarkovChain(
            step_function, distribution, number_of_steps, initial_state
        )

        myOutputRV = myMCF.getRandomVector()

        # Estimate the mean with Monte-Carlo
        sampleSize = 10000
        Y = myOutputRV.getSample(sampleSize)
        mu = Y.computeMean()[0]
        print("mu=", mu)
        relativeError = 10.0 / np.sqrt(sampleSize)
        mu_exact = 4.0
        assert_allclose(mu, mu_exact, relativeError)


if __name__ == "__main__":
    unittest.main()
