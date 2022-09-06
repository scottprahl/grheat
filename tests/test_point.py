#! /usr/bin/env python3
# pylint: disable=invalid-name
# pylint: disable=unused-variable
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long

import unittest
import numpy as np
import grheat.point_source as point

# the low level tests use functions that should not be exported.  These work
# but now that the higher level tests pass, these are skipped


class InstantaneousPoint(unittest.TestCase):

    def test_01_scalar(self):
        tp = 0
        t = 1
        T = point.instantaneous(0, 0, 0, t, 0, 0, 1, tp)

    def test_02_time_array(self):
        tp = 0
        t = np.linspace(0,10)
        T = point.instantaneous(0, 0, 0, t, 0, 0, 1, tp)

    def test_03_surface(self):
        tp = 0
        t = 1
        X, Y = np.meshgrid(np.arange(-5, 5, 1), np.arange(-5, 5, 1))
        T = point.instantaneous(X, Y, 0, t, 0, 0, 1, tp)


class ContinuousPoint(unittest.TestCase):

    def test_01_scalar(self):
        t = 1
        T = point.continuous(0, 0, 0, t, 0, 0, 1)

    def test_02_time_array(self):
        t = np.linspace(0,10)
        T = point.continuous(0, 0, 0, t, 0, 0, 1)

    def test_03_surface(self):
        tp = 0
        t = 1
        X, Y = np.meshgrid(np.arange(-5, 5, 1), np.arange(-5, 5, 1))
        T = point.continuous(X, Y, 0, t, 0, 0, 1)


class PulsedPoint(unittest.TestCase):

    def test_01_scalar(self):
        t_pulse = 0.5
        t = 1
        T = point.pulsed(0, 0, 0, t, 0, 0, 1, t_pulse)

    def test_02_time_array(self):
        t_pulse = 0.5
        t = np.linspace(0,10)
        T = point.pulsed(0, 0, 0, t, 0, 0, 1, t_pulse)

    def test_03_surface(self):
        t_pulse = 0.5
        t = 1
        X, Y = np.meshgrid(np.arange(-5, 5, 1), np.arange(-5, 5, 1))
        T = point.pulsed(X, Y, 0, t, 0, 0, 1, t_pulse)


if __name__ == '__main__':
    unittest.main()
