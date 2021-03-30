# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 14:19:18 2018

@author: Baudin

Réalise une simulation de Monte-Carlo simple sur une chaîne de Markov.

Définit la classe MarkovChain pour simplifier le traitement de 
la chaîne. 
"""

import openturns as ot
import numpy as np


class MarkovChainRandomVector(ot.PythonRandomVector):
    """Create a Markov chain random vector."""

    def __init__(self, stepFunction, stateDistr, nbSteps, initState):
        """
        Create a Markov chain random vector.

        Parameters
        ----------
        stepFunction : function
            The function which performs the step
        stateDistr : ot.Distribution
            The distribution of the state
        nbSteps : int
            The number of steps within the chain
        initState : float
            The value of the initial state

        Returns
        -------
        list
            DESCRIPTION.

        """
        self.stepFunction = stepFunction
        self.stateDistr = stateDistr
        self.nbSteps = nbSteps
        self.initState = initState
        # Nombre de variables de l'état
        nbVar = len(self.stateDistr)
        # Créée la fonction pour la chaîne
        dim = nbVar * self.nbSteps

        def myChainFunction(X):
            """Create the function across the steps."""
            Y = self.initState
            X = np.array(X)
            for i in range(self.nbSteps):
                Xn = X[i * nbVar: (i + 1) * nbVar]
                Y = self.stepFunction(Y, Xn)
            return [Y]

        self.chainfunction = ot.PythonFunction(dim, 1, myChainFunction)
        # Assemble les variables pour tous les sauts
        myVars = self.stateDistr * self.nbSteps
        myDistr = ot.ComposedDistribution(myVars)
        # Fait le lien (modele,distribution)
        myInputRV = ot.RandomVector(myDistr)
        self.randomvector = ot.CompositeRandomVector(self.chainfunction, myInputRV)
        #
        super(MarkovChainRandomVector, self).__init__(dim)
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
