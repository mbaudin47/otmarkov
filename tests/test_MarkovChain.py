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
        def myStepModel(Yn, Xn):
            """
            The function which performs the step.

            Parameters
            ----------
            Yn : ot.Point(3)
                The random input.
            Xn : ot.Point(1)
                The current state.

            Returns
            -------
            Yp : ot.Point(1)
                The new state.

            """
            P = Xn[0]
            Q = Xn[1]
            R = Xn[2]
            Yp = Yn + P * Q + R
            return Yp

        # Create the random vector.
        P = ot.Normal()
        Q = ot.Normal()
        R = ot.WeibullMin()
        stateDistr = [P, Q, R]

        # Initial state of the random vector.
        Y0 = 0.0

        # Nombre de sauts
        nbSteps = 4

        # MarkovChainFunction
        myMCF = otmarkov.MarkovChain(myStepModel, stateDistr, nbSteps, Y0)

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
