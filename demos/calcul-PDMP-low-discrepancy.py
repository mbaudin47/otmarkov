# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 16:22:30 2018

@author: Baudin

Réalise une simulation de Monte-Carlo simple sur une chaîne de Markov.

Montre comment utiliser une séquence à faible discrépance.
"""

import openturns as ot
import MarkovChain as mc

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

# MarkovChain
myMCF = mc.MarkovChain(myStepModel,stateDistr,nbSteps,Y0)
#
inputDistribution = myMCF.getInputDistribution()
modelFunction = myMCF.getFunction()
#
sampleSize = 10
ot.RandomGenerator.SetSeed(1)
myDOE = ot.MonteCarloExperiment(inputDistribution, sampleSize)
mySample = myDOE.generate()
Y = modelFunction(mySample)
print(Y)
#
inputDim = inputDistribution.getDimension()
sequence = ot.SobolSequence(inputDim)
experiment = ot.LowDiscrepancyExperiment(sequence, inputDistribution, sampleSize)
experiment.setRandomize(True)
mySample = experiment.generate()
Y = modelFunction(mySample)
print(Y)
