# -*- coding: utf-8 -*-
# Copyright 2018 - 2019 EDF.
"""
Test de la classe MarkovProcess.
"""

import openturns as ot
import unittest
from numpy.testing import assert_allclose
import otmarkov
import numpy as np


class TestMarkovProcess(unittest.TestCase):
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
            T, cumulated_time = X
            new_cumulated_time = T + cumulated_time
            return [new_cumulated_time]

        model_py = ot.PythonFunction(2, 1, single_component_model)

        # Create a parametric function from the model
        initial_state = [0.0]
        indices = [1]
        step_function = ot.ParametricFunction(model_py, indices, initial_state)

        # Evaluate step function
        cumulated_T = 0.0
        step_function.setParameter([cumulated_T])
        T = 8.0
        X = [T]
        new_state = step_function(X)
        np.testing.assert_allclose(new_state, [8.0])

        # Create the stop callback
        class StopOnHorizon():
            def __init__(self, maximum_time, verbose=False):
                self.maximum_time = maximum_time
                self.verbose = verbose
                return None

            def must_stop(self, state):
                """
                Decide whether to stop.

                The input is:
                    * state[0] : the cumulated time

                Parameters
                ----------
                state : ot.Point(1)
                    The current state.

                Returns
                -------
                stop : bool
                    True if the simulation is to be performed for another step.
                """
                cumulated_time = state[0]
                if self.verbose:
                    print("state=", state)
                    print("max. time=", self.maximum_time)
                    print("cumulated_time=", cumulated_time)
                if cumulated_time > self.maximum_time:
                    must_stop = True
                else:
                    must_stop = False
                return must_stop

        maximum_time = 20.0
        stop_callback = StopOnHorizon(maximum_time)

        # Create the random vector.
        lambda_parameter = 0.1
        dist_T = ot.Exponential(lambda_parameter)
        distribution = ot.ComposedDistribution([dist_T])

        # Create the Markov chain
        maximum_number_of_steps = 10
        markov_process = otmarkov.MarkovProcess(
            step_function, distribution, stop_callback.must_stop,
            maximum_number_of_steps, initial_state
        )
        result = markov_process.simulate()
        initial_state = result.getInitialState()
        print("initial_state=", initial_state)
        final_state = result.getFinalState()
        print("final_state=", final_state)
        history = result.getHistory()
        print("history=")
        for point in history:
            print(point)
        number_of_result_steps = result.getNumberOfSteps()
        print("number_of_steps=", number_of_result_steps)


if __name__ == "__main__":
    unittest.main()
