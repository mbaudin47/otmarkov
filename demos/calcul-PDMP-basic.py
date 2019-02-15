# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 16:22:30 2018

@author: Baudin

Réalise une simulation de Monte-Carlo simple sur une chaîne de Markov.

Utilise du Python/OpenTURNS de base.
"""

import openturns as ot

def myStepModel(Yn,Xn):
    P = Xn[0]
    Q = Xn[1]
    R = Xn[2]
    Yp = Yn + P*Q + R
    return Yp

def myChainModel(X):
    Y = 0.
    for i in range(nbSteps):
        Y = myStepModel(Y,X)
    return [Y]

# Crée les variables de l'état Xn
P = ot.Normal()
Q = ot.Normal()
R = ot.Weibull()
varOneStep = [P,Q,R]

# Assemble les variables pour tous les sauts
nbSteps = 4
myVars = [P,Q,R] * nbSteps
myDistr = ot.ComposedCopula(myVars)

# Fait le lien (modele,distribution)
nbVar = len(varOneStep)
dim = nbVar * nbSteps
model = ot.PythonFunction(dim, 1, myChainModel)
myInputRV = ot.RandomVector(myDistr)
myOutputRV = ot.RandomVector(model, myInputRV)

# Estime la moyenne par Monte-Carlo
nbSim = 1000
Y = myOutputRV.getSample(nbSim)
mu = Y.computeMean()[0]
print("Moyenne=%f" % (mu))
