#! /usr/bin/env python3
# pylint: disable=invalid-name
# pylint: disable=unused-variable
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long

import grheat
import unittest
import numpy as np

# the low level tests use functions that should not be exported.  These work
# but now that the higher level tests pass, these are skipped


class InstantaneousLine(unittest.TestCase):

    def test_01_scalar(self):
        line = grheat.Line()
        tp = 0
        t = 1
        T = line.instantaneous(0, 0, t, 0, 1, tp)

    def test_02_time_array(self):
        line = grheat.Line()
        tp = 0
        t = np.linspace(0,10)
        T = line.instantaneous(0, 0, t, 0, 1, tp)

    def test_03_surface(self):
        line = grheat.Line()
        tp = 0
        t = 1
        X, Y = np.meshgrid(np.arange(-0.005, 0.005, 0.001), np.arange(-0.005, 0.005, 0.001))
        T = line.instantaneous(X, Y, t, 0, 1, tp)


class ContinuousLine(unittest.TestCase):

    def test_01_scalar(self):
        line = grheat.Line()
        t = 1
        T = line.continuous(0, 0, t, 0, 1)

    def test_02_time_array(self):
        line = grheat.Line()
        t = np.linspace(0,10)
        T = line.continuous(0, 0, t, 0, 1)

    def test_03_surface(self):
        line = grheat.Line()
        tp = 0
        t = 1
        X, Y = np.meshgrid(np.arange(-0.005, 0.005, 0.001), np.arange(-0.005, 0.005, 0.001))
        T = line.continuous(X, Y, t, 0, 1)


class PulsedLine(unittest.TestCase):

    def test_01_scalar(self):
        line = grheat.Line()
        t_pulse = 0.5
        t = 1
        yp = 0.001  # meters
        T = line.pulsed(0, 0, t, 0, yp, t_pulse)

    def test_02_time_array(self):
        line = grheat.Line()
        t_pulse = 0.5
        t = np.linspace(0,10)
        yp = 0.001  # meters
        T = line.pulsed(0, 0, t, 0, yp, t_pulse)

    def test_03_surface(self):
        line = grheat.Line()
        t_pulse = 0.5
        t = 1
        yp = 0.001  # meters
        X, Y = np.meshgrid(np.arange(-0.005, 0.005, 0.001), np.arange(-0.005, 0.005, 0.001))
        T = line.pulsed(X, Y, t, 0, yp, t_pulse)

class InstantVsPulsed(unittest.TestCase):

    def test_01_instant(self):
        """Short pulse result should be same as instantaneous source."""
        line = grheat.Line()
        t_pulse = 0.00001
        tp = 0
        t = 1
        yp = 0.001  # meters
        T1 = line.instantaneous(0, 0, t, 0, yp, tp)
        T2 = line.pulsed(0, 0, t, 0, yp, t_pulse)
        self.assertAlmostEqual(T1, T2, delta=0.001)

    def test_02_instant(self):
        """Short pulse result should be same as instantaneous source."""
        line = grheat.Line()
        t_pulse = 0.00001
        tp = 0
        t = np.linspace(0,10)
        yp = 0.001  # meters
        T1 = line.instantaneous(0, 0, t, 0, yp, tp)
        T2 = line.pulsed(0, 0, t, 0, yp, t_pulse)
        self.assertAlmostEqual(T1[3], T2[3], delta=0.001)
        self.assertAlmostEqual(T1[13], T2[13], delta=0.001)

    def test_03_instant(self):
        """Short pulse result should be same as instantaneous source."""
        line = grheat.Line()
        t_pulse = 0.00001
        tp = 0
        t = 1
        yp = 0.001  # meters
        X, Y = np.meshgrid(np.arange(-0.005, 0.005, 0.001), np.arange(-0.005, 0.005, 0.001))

        T1 = line.instantaneous(X, Y, t, 0, yp, tp)
        T2 = line.pulsed(X, Y, t, 0, yp, t_pulse)
        self.assertAlmostEqual(T1[0,3], T2[0,3], delta=0.001)
        self.assertAlmostEqual(T1[3,1], T2[3,1], delta=0.001)

if __name__ == '__main__':
    unittest.main()
