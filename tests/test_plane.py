#! /usr/bin/env python3
# pylint: disable=invalid-name
# pylint: disable=unused-variable
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=plane-too-long

import unittest
import numpy as np
import grheat

class InstantaneousPlane(unittest.TestCase):

    def test_01_scalar(self):
        plane = grheat.Plane()
        tp = 0
        t = 1
        zp = 0.001  # meters
        T = plane.instantaneous(0, t, zp, tp)

    def test_02_time_array(self):
        plane = grheat.Plane()
        tp = 0
        zp = 0.001  # meters
        t = np.linspace(0,10)
        T = plane.instantaneous(0, t, zp, tp)


class ContinuousPlane(unittest.TestCase):

    def test_01_scalar(self):
        plane = grheat.Plane()
        t = 1
        zp = 0.001  # meters
        T = plane.continuous(0, t, zp)

    def test_02_time_array(self):
        plane = grheat.Plane()
        t = np.linspace(0,10)
        zp = 0.001  # meters
        T = plane.continuous(0, t, zp)


class PulsedPlane(unittest.TestCase):

    def test_01_scalar(self):
        plane = grheat.Plane()
        t_pulse = 0.5
        t = 1
        zp = 0.001  # meters
        T = plane.pulsed(0, t, zp, t_pulse)

    def test_02_time_array(self):
        plane = grheat.Plane()
        t_pulse = 0.5
        t = np.linspace(0,10)
        zp = 0.001  # meters
        T = plane.pulsed(0, t, zp, t_pulse)

class InstantVsPulsed(unittest.TestCase):

    def test_01_instant(self):
        """Short pulse result should be same as instantaneous source."""
        plane = grheat.Plane()
        t_pulse = 0.00001
        tp = 0
        t = 1
        zp = 0.001  # meters
        radiant_exposure = 1/1e-6   # 1 J/mm²
        T1 = radiant_exposure * plane.instantaneous(0, t, zp, tp)
        T2 = radiant_exposure * plane.pulsed(0, t, zp, t_pulse)
        self.assertAlmostEqual(T1, T2, delta=0.001)

    def test_02_instant(self):
        """Short pulse result should be same as instantaneous source."""
        plane = grheat.Plane()
        t_pulse = 0.00001
        tp = 0
        t = np.linspace(0,10)
        zp = 0.001  # meters
        radiant_exposure = 1/1e-6   # 1 J/mm²
        T1 = radiant_exposure * plane.instantaneous(0, t, zp, tp)
        T2 = radiant_exposure * plane.pulsed(0, t, zp, t_pulse)
        self.assertAlmostEqual(T1[3], T2[3], delta=0.001)
        self.assertAlmostEqual(T1[13], T2[13], delta=0.001)

if __name__ == '__main__':
    unittest.main()
