# -*- coding: utf-8 -*-
# Copyright 2018 - 2019 EDF.
"""
Test de la classe MarkovChainRandomVector.
"""

import openturns as ot
import unittest
from numpy.testing import assert_allclose
import otmarkov
import numpy as np


class TestMarkovChainRandomVector(unittest.TestCase):
    def test_PQR(self):
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
        initial_state = ot.Point([0.0])
        indices = [3]
        step_function = ot.ParametricFunction(model_py, indices, initial_state)

        # Create the distribution of the random input.
        P = ot.Normal()
        Q = ot.Normal()
        R = ot.WeibullMin()
        distribution = ot.ComposedDistribution([P, Q, R])

        # Create the Markov chain
        number_of_steps = 4
        mc_random_vector = otmarkov.MarkovChainRandomVector(
            step_function, distribution, number_of_steps, initial_state
        )
        random_vector = ot.RandomVector(mc_random_vector)

        # Test getRealization
        y = random_vector.getRealization()[0]
        n = ot.Normal(4.0, 2.83)
        ninterval = n.computeBilateralConfidenceInterval(1.0 - 1.0e-6)
        a = ninterval.getLowerBound()[0]
        b = ninterval.getUpperBound()[0]
        self.assertTrue((y > a) & (y < b))

        # Create a sample from the Markov chain
        sampleSize = 1000
        sample = random_vector.getSample(sampleSize)
        sample_mean = sample.computeMean()[0]
        relativeError = 10.0 / np.sqrt(sampleSize)
        print("sample_mean=", sample_mean)
        mu_exact = 4.0
        assert_allclose(sample_mean, mu_exact, relativeError)

    def test_SingleComponent(self):
        def single_component_model(X):
            """
            Simulate a single-component system.

            The inputs are:
                * X[0] : T, the life time of the component at this step
                * X[1] : cumulated_T, the cumulate life time of the component

            The output is the new state, which is the sum of the cumulated life time
            and the current life time.

            Parameters
            ----------
            X : ot.Point(2)
                The input of the model.

            Returns
            -------
            new_cumulated_T : ot.Point(1)
                The updated cumulated life time of the component.
            """
            X = ot.Point(X)
            T, cumulated_T = X
            new_cumulated_T = T + cumulated_T
            return [new_cumulated_T]

        model_py = ot.PythonFunction(2, 1, single_component_model)

        # Create a parametric function from the model
        initial_state = ot.Point([0.0])
        indices = [1]
        step_function = ot.ParametricFunction(model_py, indices, initial_state)

        # Evaluate step function
        cumulated_T = 0.0
        step_function.setParameter([cumulated_T])
        T = 8.0
        X = [T]
        new_state = step_function(X)
        np.testing.assert_allclose(new_state, [8.0])

        # Create the random vector.
        lambda_parameter = 0.1
        dist_T = ot.Exponential(lambda_parameter)
        distribution = ot.ComposedDistribution([dist_T])

        # Create the Markov chain
        number_of_steps = 4
        mc_random_vector = otmarkov.MarkovChainRandomVector(
            step_function, distribution, number_of_steps, initial_state
        )
        random_vector = ot.RandomVector(mc_random_vector)

        # Estimate the mean with Monte-Carlo
        sampleSize = 1000
        sample = random_vector.getSample(sampleSize)
        sample_mean = sample.computeMean()[0]
        print("sample_mean=", sample_mean)
        relativeError = 10.0 / np.sqrt(sampleSize)
        gamma = ot.Gamma(number_of_steps, lambda_parameter)
        mu_exact = gamma.getMean()
        assert_allclose(sample_mean, mu_exact, relativeError)


if __name__ == "__main__":
    unittest.main()
