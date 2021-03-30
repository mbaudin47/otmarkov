# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 16:22:30 2018

@author: Baudin

Réalise une simulation de Monte-Carlo simple sur une chaîne de Markov.

Montre comment utiliser une séquence à faible discrépance.
"""

import openturns as ot
import otmarkov

# ot.RandomGenerator.SetSeed(0)


def step_function(state, X):
    """
    Perform one step.

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
    P, Q, R = X
    new_state = state + P * Q + R
    return new_state


# Crée les variables de l'état Xn
P = ot.Normal()
Q = ot.Normal()
R = ot.WeibullMin()
distribution = ot.ComposedDistribution([P, Q, R])

# Etat initial
Y0 = 0.0

# Nombre de sauts
nbSteps = 4

# MarkovChain
markov_chain = otmarkov.MarkovChain(step_function, distribution, nbSteps, Y0)
#
inputDistribution = markov_chain.getInputDistribution()
modelFunction = markov_chain.getFunction()
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
