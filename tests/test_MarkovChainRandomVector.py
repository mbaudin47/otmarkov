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
        # ot.RandomGenerator.SetSeed(0)
        def step_function(Yn, Xn):
            P = Xn[0]
            Q = Xn[1]
            R = Xn[2]
            Yp = Yn + P * Q + R
            return Yp

        # Create the distribution of the random input.
        P = ot.Normal()
        Q = ot.Normal()
        R = ot.WeibullMin()
        distribution = ot.ComposedDistribution([P, Q, R])

        initial_state = 0.0

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

        # Test getSample
        # Estime la moyenne par Monte-Carlo simple
        sampleSize = 100000
        Y = myMCF.getSample(sampleSize)
        mu = Y.computeMean()[0]
        relativeError = 10 * 2.8 / np.sqrt(sampleSize) / 4.0
        mu_exact = 4.0
        assert_allclose(mu, mu_exact, relativeError)


if __name__ == "__main__":
    unittest.main()
