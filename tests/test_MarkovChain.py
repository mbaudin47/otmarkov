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
        # ot.RandomGenerator.SetSeed(0)
        def step_function(state, X):
            """
            The function which performs the step.

            Parameters
            ----------
            state : ot.Point(1)
                The current state.
            X : ot.Point(3)
                The random input.

            Returns
            -------
            new_state : ot.Point(1)
                The new state.

            """
            P = X[0]
            Q = X[1]
            R = X[2]
            new_state = state + P * Q + R
            return new_state

        # Create the random vector.
        P = ot.Normal()
        Q = ot.Normal()
        R = ot.WeibullMin()
        distribution = ot.ComposedDistribution([P, Q, R])

        # Initial state of the chain.
        initial_state = 0.0

        number_of_steps = 4

        myMCF = otmarkov.MarkovChain(step_function, distribution, number_of_steps,
                                     initial_state)

        myOutputRV = myMCF.getOutputRandomVector()

        # Estime la moyenne par Monte-Carlo
        sampleSize = 100000
        Y = myOutputRV.getSample(sampleSize)
        mu = Y.computeMean()[0]
        relativeError = 10 * 2.8 / np.sqrt(sampleSize) / 4.0
        mu_exact = 4.0
        assert_allclose(mu, mu_exact, relativeError)


if __name__ == "__main__":
    unittest.main()
