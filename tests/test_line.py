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

class InstantaneousLine(unittest.TestCase):

    def test_01_scalar(self):
        xp, yp = 0, 0.001         # meters
        line = grheat.Line(xp, yp)
        tp = 0
        t = 1
        T = line.instantaneous(0, 0, t, tp)

    def test_02_time_array(self):
        xp, yp = 0, 0.001         # meters
        line = grheat.Line(xp, yp)
        tp = 0
        t = np.linspace(0,10)
        T = line.instantaneous(0, 0, t, tp)

    def test_03_surface(self):
        xp, yp = 0, 0.001         # meters
        line = grheat.Line(xp, yp)
        tp = 0
        t = 1
        X, Y = np.meshgrid(np.arange(-0.0051, 0.005, 0.001), np.arange(-0.005, 0.005, 0.001))
        T = line.instantaneous(X, Y, t, tp)


class ContinuousLine(unittest.TestCase):

    def test_01_scalar(self):
        xp, yp = 0, 0.001         # meters
        line = grheat.Line(xp, yp)
        t = 1
        T = line.continuous(0, 0, t)

    def test_02_time_array(self):
        xp, yp = 0, 0.001         # meters
        line = grheat.Line(xp, yp)
        t = np.linspace(0,10)
        T = line.continuous(0, 0, t)

    def test_03_surface(self):
        xp, yp = 0, 0.001         # meters
        line = grheat.Line(xp, yp)
        tp = 0
        t = 1
        X, Y = np.meshgrid(np.arange(-0.0051, 0.005, 0.001), np.arange(-0.005, 0.005, 0.001))
        T = line.continuous(X, Y, t)


class PulsedLine(unittest.TestCase):

    def test_01_scalar(self):
        xp, yp = 0, 0.001         # meters
        line = grheat.Line(xp, yp)
        t_pulse = 0.5
        t = 1
        T = line.pulsed(0, 0, t, t_pulse)

    def test_02_time_array(self):
        xp, yp = 0, 0.001         # meters
        line = grheat.Line(xp, yp)
        t_pulse = 0.5
        t = np.linspace(0,10)
        T = line.pulsed(0, 0, t, t_pulse)

    def test_03_surface(self):
        xp, yp = 0, 0.001         # meters
        line = grheat.Line(xp, yp)
        t_pulse = 0.5
        t = 1
        X, Y = np.meshgrid(np.arange(-0.0051, 0.005, 0.001), np.arange(-0.005, 0.005, 0.001))
        T = line.pulsed(X, Y, t, t_pulse)

class InstantVsPulsed(unittest.TestCase):

    def test_01_instant(self):
        """Short pulse result should be same as instantaneous source."""
        xp, yp = 0, 0.001         # meters
        line = grheat.Line(xp, yp)
        t_pulse = 0.00001
        tp = 0
        t = 1
        T1 = line.instantaneous(0, 0, t, tp)
        T2 = line.pulsed(0, 0, t, t_pulse)
        self.assertAlmostEqual(T1, T2, delta=0.001)

    def test_02_instant(self):
        """Short pulse result should be same as instantaneous source."""
        xp, yp = 0, 0.001         # meters
        line = grheat.Line(xp, yp)
        t_pulse = 0.00001
        tp = 0
        t = np.linspace(0,10)
        T1 = line.instantaneous(0, 0, t, tp)
        T2 = line.pulsed(0, 0, t, t_pulse)
        self.assertAlmostEqual(T1[3], T2[3], delta=0.001)
        self.assertAlmostEqual(T1[13], T2[13], delta=0.001)

    def test_03_instant(self):
        """Short pulse result should be same as instantaneous source."""
        xp, yp = 0, 0.001         # meters
        line = grheat.Line(xp, yp)
        t_pulse = 0.00001
        tp = 0
        t = 1
        X, Y = np.meshgrid(np.arange(-0.0051, 0.005, 0.001), np.arange(-0.005, 0.005, 0.001))

        T1 = line.instantaneous(X, Y, t, tp)
        T2 = line.pulsed(X, Y, t, t_pulse)
        self.assertAlmostEqual(T1[0,3], T2[0,3], delta=0.001)
        self.assertAlmostEqual(T1[3,1], T2[3,1], delta=0.001)

if __name__ == '__main__':
    unittest.main()
