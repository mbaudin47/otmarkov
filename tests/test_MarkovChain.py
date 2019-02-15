# -*- coding: utf-8 -*-
# Copyright 2018 - 2019 EDF.
"""
Test de la classe MarkovChain.
"""

import openturns as ot
import unittest
from numpy.testing import assert_allclose
import MarkovChain as mc
import numpy as np

class TestMarkovChain(unittest.TestCase):
    def test_PQR(self):
        #ot.RandomGenerator.SetSeed(0)
        def myStepModel(Yn,Xn):
            P = Xn[0]
            Q = Xn[1]
            R = Xn[2]
            Yp = Yn + P*Q + R
            return Yp
        
        # Crée les variables de l'état Xn
        P = ot.Normal()
        Q = ot.Normal()
        R = ot.Weibull()
        stateDistr = [P,Q,R]
        
        # Etat initial
        Y0 = 0.
        
        # Nombre de sauts
        nbSteps = 4
        
        # MarkovChainFunction
        myMCF = mc.MarkovChain()
        myMCF.setStepFunction(myStepModel)
        myMCF.setRandomStateDistribution(stateDistr)
        myMCF.setNumberOfSteps(nbSteps)
        myMCF.setDeterministicInitialState(Y0)
        
        myOutputRV = myMCF.getOutputRandomVector()
        
        # Estime la moyenne par Monte-Carlo
        sampleSize = 100000
        Y = myOutputRV.getSample(sampleSize)
        mu = Y.computeMean()[0]
        relativeError =  10 * 2.8/np.sqrt(sampleSize)/4.
        mu_exact = 4.0
        assert_allclose(mu,mu_exact,relativeError)

if __name__ == '__main__':
    unittest.main()
