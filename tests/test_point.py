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
        zp = 0.001  # meters
        T = point.pulsed(0, 0, 0, t, 0, 0, zp, t_pulse)

    def test_02_time_array(self):
        t_pulse = 0.5
        t = np.linspace(0,10)
        zp = 0.001  # meters
        T = point.pulsed(0, 0, 0, t, 0, 0, zp, t_pulse)

    def test_03_surface(self):
        t_pulse = 0.5
        t = 1
        zp = 0.001  # meters
        X, Y = np.meshgrid(np.arange(-0.005, 0.005, 0.001), np.arange(-0.005, 0.005, 0.001))
        T = point.pulsed(X, Y, 0, t, 0, 0, zp, t_pulse)

class InstantVsPulsed(unittest.TestCase):

    def test_01_instant(self):
        """Short pulse result should be same as instantaneous source."""
        t_pulse = 0.00001
        tp = 0
        t = 1
        zp = 0.001  # meters
        T1 = point.instantaneous(0, 0, 0, t, 0, 0, zp, tp)
        T2 = point.pulsed(0, 0, 0, t, 0, 0, zp, t_pulse)
        self.assertAlmostEqual(T1, T2, delta=0.001)

    def test_02_instant(self):
        """Short pulse result should be same as instantaneous source."""
        t_pulse = 0.00001
        tp = 0
        t = np.linspace(0,10)
        zp = 0.001  # meters
        T1 = point.instantaneous(0, 0, 0, t, 0, 0, zp, tp)
        T2 = point.pulsed(0, 0, 0, t, 0, 0, zp, t_pulse)
        self.assertAlmostEqual(T1[3], T2[3], delta=0.001)
        self.assertAlmostEqual(T1[13], T2[13], delta=0.001)

    def test_03_instant(self):
        """Short pulse result should be same as instantaneous source."""
        t_pulse = 0.00001
        tp = 0
        t = 1
        zp = 0.001  # meters
        X, Y = np.meshgrid(np.arange(-0.005, 0.005, 0.001), np.arange(-0.005, 0.005, 0.001))

        T1 = point.instantaneous(X, Y, 0, t, 0, 0, zp, tp)
        T2 = point.pulsed(X, Y, 0, t, 0, 0, zp, t_pulse)
        self.assertAlmostEqual(T1[0,3], T2[0,3], delta=0.001)
        self.assertAlmostEqual(T1[3,1], T2[3,1], delta=0.001)

if __name__ == '__main__':
    unittest.main()
