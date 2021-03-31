# -*- coding: utf-8 -*-
# Copyright 2018 - 2019 EDF.
"""
Test de la classe MarkovChainRandomVector.
"""

import openturns as ot
import unittest
from numpy.testing import assert_allclose
import otmarkov
import numpy as np


class TestMarkovChainRandomVector(unittest.TestCase):
    def test_PQR(self):
        def model(X):
            """
            The function which performs the step.

            The inputs are:
                * X[0] : P
                * X[1] : Q
                * X[2] : R
                * X[3] : state

            The output is the new state.

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

        # Create the distribution of the random input.
        P = ot.Normal()
        Q = ot.Normal()
        R = ot.WeibullMin()
        distribution = ot.ComposedDistribution([P, Q, R])

        # Create the Markov chain
        number_of_steps = 4
        myMCF = otmarkov.MarkovChainRandomVector(
            step_function, distribution, number_of_steps, initial_state
        )

        # Test getRealization
        y = myMCF.getRealization()[0]
        n = ot.Normal(4, 2.83)
        ninterval = n.computeBilateralConfidenceInterval(1 - 1.0e-6)
        a = ninterval.getLowerBound()[0]
        b = ninterval.getUpperBound()[0]
        self.assertTrue((y > a) & (y < b))

        # Create a sample from the Markov chain
        sampleSize = 1000
        Y = myMCF.getSample(sampleSize)
        mu = Y.computeMean()[0]
        relativeError = 10.0 / np.sqrt(sampleSize)
        mu_exact = 4.0
        assert_allclose(mu, mu_exact, relativeError)


if __name__ == "__main__":
    unittest.main()
