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

class MarkovChain:
    def __init__(self,stepFunction,stateDistr,nbSteps,initState):
        '''
        Creates a new Markov Chain
 
        stepFunction : a function, the function which performs the step
        stateDistr : a Distribution, the distribution of the state
        nbSteps : an integer, the number of steps within the chain
        initState : a double, the value of the initial state
        '''
        self.stepFunction = stepFunction
        self.stateDistr = stateDistr
        self.nbSteps = nbSteps
        self.initState = initState

    def getInputDistribution(self):
        # Assemble les variables pour tous les sauts
        myVars = self.stateDistr * self.nbSteps
        myDistr = ot.ComposedCopula(myVars)
        return myDistr

    def getFunction(self):
        def myChainFunction(X):
            Y = self.initState
            X = np.array(X)
            for i in range(self.nbSteps):
                Xn = X[i*nbVar:(i+1)*nbVar]
                Y = self.stepFunction(Y,Xn)
            return [Y]
        # Nombre de variables de l'état
        nbVar = len(self.stateDistr)
        # Créée la fonction pour la chaîne
        dim = nbVar * self.nbSteps
        model = ot.PythonFunction(dim, 1, myChainFunction)
        return model

    def getOutputRandomVector(self):
        # Assemble les variables pour tous les sauts
        myDistr = self.getInputDistribution()
        # Fait le lien (modele,distribution)
        model = self.getFunction()
        myInputRV = ot.RandomVector(myDistr)
        myOutputRV = ot.RandomVector(model, myInputRV)
        return myOutputRV
    
