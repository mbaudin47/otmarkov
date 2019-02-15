# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 14:19:18 2018

@author: Baudin

Calcule la moyenne exacte d'une fonction de 3 variables aléatoires.
"""

import openturns as ot

P = ot.Normal()
Q = ot.Normal()
R = ot.Weibull()

F = P*Q + R + P*Q + R + P*Q + R + P*Q + R
mu = F.getMean()[0]
print("Mean=%f" % (mu))
sigma = F.getStandardDeviation()[0]
print("Mean=%f" % (mu))
