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
        yp, zp = 0, 0.001         # meters
        line = grheat.Line(yp, zp)
        tp = 0
        t = 1
        T = line.instantaneous(0, 0, t, tp)

    def test_02_time_array(self):
        yp, zp = 0, 0.001         # meters
        line = grheat.Line(yp, zp)
        tp = 0
        t = np.linspace(0, 10)
        T = line.instantaneous(0, 0, t, tp)

    def test_03_surface(self):
        yp, zp = 0, 0.001         # meters
        line = grheat.Line(yp, zp)
        tp = 0
        t = 1
        Y, Z = np.meshgrid(np.arange(-0.0051, 0.005, 0.001), np.arange(-0.005, 0.005, 0.001))
        T = line.instantaneous(Y, Z, t, tp)


class ContinuousLine(unittest.TestCase):

    def test_01_scalar(self):
        yp, zp = 0, 0.001         # meters
        line = grheat.Line(yp, zp)
        t = 1
        T = line.continuous(0, 0, t)

    def test_02_time_array(self):
        yp, zp = 0, 0.001         # meters
        line = grheat.Line(yp, zp)
        t = np.linspace(0, 10)
        T = line.continuous(0, 0, t)

    def test_03_surface(self):
        yp, zp = 0, 0.001         # meters
        line = grheat.Line(yp, zp)
        tp = 0
        t = 1
        Y, Z = np.meshgrid(np.arange(-0.0051, 0.005, 0.001), np.arange(-0.005, 0.005, 0.001))
        T = line.continuous(Y, Z, t)


class PulsedLine(unittest.TestCase):

    def test_01_scalar(self):
        yp, zp = 0, 0.001         # meters
        line = grheat.Line(yp, zp)
        t_pulse = 0.5
        t = 1
        T = line.pulsed(0, 0, t, t_pulse)

    def test_02_time_array(self):
        yp, zp = 0, 0.001         # meters
        line = grheat.Line(yp, zp)
        t_pulse = 0.5
        t = np.linspace(0, 10)
        T = line.pulsed(0, 0, t, t_pulse)

    def test_03_surface(self):
        yp, zp = 0, 0.001         # meters
        line = grheat.Line(yp, zp)
        t_pulse = 0.5
        t = 1
        Y, Z = np.meshgrid(np.arange(-0.0051, 0.005, 0.001), np.arange(-0.005, 0.005, 0.001))
        T = line.pulsed(Y, Z, t, t_pulse)


class InstantVsPulsed(unittest.TestCase):

    def test_01_instant(self):
        """Short pulse result should be same as instantaneous source."""
        yp, zp = 0, 0.001         # meters
        line = grheat.Line(yp, zp)
        t_pulse = 0.00001
        tp = 0
        t = 1
        T1 = line.instantaneous(0, 0, t, tp)
        T2 = line.pulsed(0, 0, t, t_pulse)
        self.assertAlmostEqual(T1, T2, delta=0.001)

    def test_02_instant(self):
        """Short pulse result should be same as instantaneous source."""
        yp, zp = 0, 0.001         # meters
        line = grheat.Line(yp, zp)
        t_pulse = 0.00001
        tp = 0
        t = np.linspace(0, 10)
        T1 = line.instantaneous(0, 0, t, tp)
        T2 = line.pulsed(0, 0, t, t_pulse)
        self.assertAlmostEqual(T1[3], T2[3], delta=0.001)
        self.assertAlmostEqual(T1[13], T2[13], delta=0.001)

    def test_03_instant(self):
        """Short pulse result should be same as instantaneous source."""
        yp, zp = 0, 0.001         # meters
        line = grheat.Line(yp, zp)
        t_pulse = 0.00001
        tp = 0
        t = 1
        Y, Z = np.meshgrid(np.arange(-0.0051, 0.005, 0.001), np.arange(-0.005, 0.005, 0.001))

        T1 = line.instantaneous(Y, Z, t, tp)
        T2 = line.pulsed(Y, Z, t, t_pulse)
        self.assertAlmostEqual(T1[0, 3], T2[0, 3], delta=0.001)
        self.assertAlmostEqual(T1[3, 1], T2[3, 1], delta=0.001)


class ConstantBoundary(unittest.TestCase):

    def test_01_zero(self):
        """Surface temperature should be zero."""
        yp, zp = 0.0001, 0.0001     # meters
        line = grheat.Line(yp, zp, boundary='zero')
        t_pulse = 1
        t = 2
        T = line.pulsed(0.0001, 0, t, t_pulse)
        self.assertEqual(T, 0)

    def test_02_zero(self):
        """Surface temperature should be zero at all times."""
        yp, zp = 0.0001, 0.0001     # meters
        line = grheat.Line(yp, zp, boundary='zero')
        t_pulse = 1
        t = np.linspace(0, 10)
        T = line.pulsed(0.0001, 0, t, t_pulse)
        self.assertEqual(T[3], 0)
        self.assertEqual(T[13], 0)


class AdiabaticBoundary(unittest.TestCase):

    def test_01_adiabatic(self):
        """Temperature should be equal above and below."""
        yp, zp = 0.0001, 0.0001     # meters
        line = grheat.Line(yp, zp, boundary='adiabatic')
        t_pulse = 1
        t = 2
        T1 = line.pulsed(0, +0.0001, t, t_pulse)
        T2 = line.pulsed(0, -0.0001, t, t_pulse)
        self.assertAlmostEqual(T1, T2, delta=1e-8)

    def test_02_adiabatic(self):
        """Temperature should be equal above and below at all times."""
        yp, zp = 0.0001, 0.0001     # meters
        line = grheat.Line(yp, zp, boundary='adiabatic')
        t_pulse = 1
        t = np.linspace(0, 2)
        T1 = line.pulsed(0, +0.0001, t, t_pulse)
        T2 = line.pulsed(0, -0.0001, t, t_pulse)
        self.assertAlmostEqual(T1[3], T2[3], delta=1e-8)
        self.assertAlmostEqual(T1[13], T2[13], delta=1e-8)


if __name__ == '__main__':
    unittest.main()
