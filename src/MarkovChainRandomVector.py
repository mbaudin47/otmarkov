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
    def __init__(self,stepFunction,stateDistr,nbSteps,initState):
        self.stepFunction = stepFunction
        self.stateDistr = stateDistr
        self.nbSteps = nbSteps
        self.initState = initState
        # Nombre de variables de l'état
        nbVar = len(self.stateDistr)
        # Créée la fonction pour la chaîne
        dim = nbVar * self.nbSteps
        # Créée la fonction de transition
        def myChainFunction(X):
            Y = self.initState
            X = np.array(X)
            for i in range(self.nbSteps):
                Xn = X[i*nbVar:(i+1)*nbVar]
                Y = self.stepFunction(Y,Xn)
            return [Y]
        self.chainfunction = ot.PythonFunction(dim, 1, myChainFunction)
        # Assemble les variables pour tous les sauts
        myVars = self.stateDistr * self.nbSteps
        myDistr = ot.ComposedCopula(myVars)
        # Fait le lien (modele,distribution)
        myInputRV = ot.RandomVector(myDistr)
        self.randomvector = ot.RandomVector(self.chainfunction, myInputRV)
        #
        super(MarkovChainRandomVector, self).__init__(dim)

    def getRealization(self):
        return self.randomvector.getRealization()

    def getSample(self, size):
        return self.randomvector.getSample(size)


