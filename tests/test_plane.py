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
        zp = 0.001  # meters
        tp = 0
        t = 1
        plane = grheat.Plane(zp)
        T = plane.instantaneous(0, t, tp)

    def test_02_time_array(self):
        zp = 0.001  # meters
        tp = 0
        t = np.linspace(0,10)
        plane = grheat.Plane(zp)
        T = plane.instantaneous(0, t, tp)


class ContinuousPlane(unittest.TestCase):

    def test_01_scalar(self):
        zp = 0.001  # meters
        t = 1
        plane = grheat.Plane(zp)
        T = plane.continuous(0, t)

    def test_02_time_array(self):
        zp = 0.001  # meters
        t = np.linspace(0,10)
        plane = grheat.Plane(zp)
        T = plane.continuous(0, t)


class PulsedPlane(unittest.TestCase):

    def test_01_scalar(self):
        zp = 0.001  # meters
        t_pulse = 0.5
        t = 1
        plane = grheat.Plane(zp)
        T = plane.pulsed(0, t, t_pulse)

    def test_02_time_array(self):
        zp = 0.001  # meters
        t_pulse = 0.5
        t = np.linspace(0,10)
        plane = grheat.Plane(zp)
        T = plane.pulsed(0, t, t_pulse)

class InstantVsPulsed(unittest.TestCase):

    def test_01_instant(self):
        """Short pulse result should be same as instantaneous source."""
        zp = 0.001  # meters
        t_pulse = 0.00001
        tp = 0
        t = 1
        radiant_exposure = 1e6    # 1 J/mm² in J/m²
        plane = grheat.Plane(zp)
        T1 = radiant_exposure * plane.instantaneous(0, t, tp)
        T2 = radiant_exposure * plane.pulsed(0, t, t_pulse)
        self.assertAlmostEqual(T1, T2, delta=0.001)

    def test_02_instant(self):
        """Short pulse result should be same as instantaneous source."""
        zp = 0.001  # meters
        t_pulse = 0.00001
        tp = 0
        t = np.linspace(0,10)
        radiant_exposure = 1e6    # 1 J/mm² in J/m²
        plane = grheat.Plane(zp)
        T1 = radiant_exposure * plane.instantaneous(0, t, tp)
        T2 = radiant_exposure * plane.pulsed(0, t, t_pulse)
        self.assertAlmostEqual(T1[3], T2[3], delta=0.001)
        self.assertAlmostEqual(T1[13], T2[13], delta=0.001)

if __name__ == '__main__':
    unittest.main()
