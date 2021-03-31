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


def model(X):
    """
    The function which performs the step.

    The inputs are:
        * X[0] : P
        * X[1] : Q
        * X[2] : R
        * X[3] : state

    The output is the new state.

    Parameters
    ----------
    X : ot.Point(4)
        The input of the model.

    Returns
    -------
    new_state : ot.Point(1)
        The new state.
    """
    P = X[0]
    Q = X[1]
    R = X[2]
    state = X[3]
    new_state = state + P * Q + R
    return [new_state]


model_py = ot.PythonFunction(4, 1, model)

# Create a parametric function from the model
# The input is random, the parameter is the state, the output is the new state.
initial_state = [0.0]
indices = [3]
step_function = ot.ParametricFunction(model_py, indices, initial_state)


# Crée les variables de l'état Xn
P = ot.Normal()
Q = ot.Normal()
R = ot.WeibullMin()
distribution = ot.ComposedDistribution([P, Q, R])

# Nombre de sauts
nbSteps = 4

# MarkovChain
markov_chain = otmarkov.MarkovChain(step_function, distribution, nbSteps, initial_state)
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
