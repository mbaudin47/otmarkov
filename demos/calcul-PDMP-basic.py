# -*- coding: utf-8 -*-
"""
@author: Baudin

Réalise une simulation de Monte-Carlo simple sur une chaîne de Markov.

Utilise du Python/OpenTURNS de base.
"""

import openturns as ot


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


def myChainModel(X):
    """Compute Markov chain output."""
    X = ot.Point(X)
    Y = 0.0
    input_step_dimension = 3
    for i in range(nbSteps):
        index_start = i * input_step_dimension
        index_stop = (i + 1) * input_step_dimension
        Xn = X[index_start:index_stop]
        Y = step_function(Y, Xn)
    return [Y]


# Crée les variables de l'état Xn
P = ot.Normal()
Q = ot.Normal()
R = ot.WeibullMin()
distribution = ot.ComposedDistribution([P, Q, R])

# Assemble les variables pour tous les sauts
nbSteps = 4
myVars = [distribution] * nbSteps
myDistr = ot.BlockIndependentDistribution(myVars)

# Fait le lien (modele,distribution)
nbVar = distribution.getDimension()
dim = nbVar * nbSteps
model = ot.PythonFunction(dim, 1, myChainModel)
myInputRV = ot.RandomVector(myDistr)
myOutputRV = ot.CompositeRandomVector(model, myInputRV)

# Estime la moyenne par Monte-Carlo
nbSim = 1000
Y = myOutputRV.getSample(nbSim)
mu = Y.computeMean()[0]
print("Moyenne=%f" % (mu))
