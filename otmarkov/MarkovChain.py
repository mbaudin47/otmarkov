# -*- coding: utf-8 -*-
"""
@author: Baudin

Réalise une simulation de Monte-Carlo simple sur une chaîne de Markov.

Définit la classe MarkovChain pour qui implémente une chaîne de Markov à
sauts discrets.
"""

import openturns as ot
import numpy as np


class MarkovChain:
    """A Markov chain class."""

    def __init__(self, stepFunction, stateDistr, nbSteps, initState):
        """
        Create a new Markov Chain.

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
        """
        self.stepFunction = stepFunction
        self.stateDistr = stateDistr
        self.nbSteps = nbSteps
        self.initState = initState

    def getInputDistribution(self):
        """
        Return the input distribution.

        Returns
        -------
        myDistr : ot.Distribution
            The distribution of the random state, for all steps.

        """
        # Assemble les variables pour tous les sauts
        myVars = self.stateDistr * self.nbSteps
        myDistr = ot.ComposedDistribution(myVars)
        return myDistr

    def getFunction(self):
        """
        Return the function for all steps.

        This function takes the initial state as input
        and returns the final state as output.
        It performs a loop over the number of steps.
        At each step, the new state is computed from the curent
        random vector and the old step.

        Returns
        -------
        model: ot.PythonFunction
            The model for all steps.

        """
        def myChainFunction(X):
            Y = self.initState
            X = np.array(X)
            for i in range(self.nbSteps):
                Xn = X[i * nbVar: (i + 1) * nbVar]
                Y = self.stepFunction(Y, Xn)
            return [Y]

        # Number of variables of the state.
        nbVar = len(self.stateDistr)
        # Créée la fonction pour la chaîne
        dim = nbVar * self.nbSteps
        model = ot.PythonFunction(dim, 1, myChainFunction)
        return model

    def getOutputRandomVector(self):
        """
        Return the output random vector.

        Returns
        -------
        myOutputRV : ot.CompositeRandomVector
            The random vector which the output of the Markov chain.

        """
        # Assemble les variables pour tous les sauts
        myDistr = self.getInputDistribution()
        # Fait le lien (modele,distribution)
        model = self.getFunction()
        myInputRV = ot.RandomVector(myDistr)
        myOutputRV = ot.CompositeRandomVector(model, myInputRV)
        return myOutputRV
