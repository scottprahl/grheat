#! /usr/bin/env python3
# pylint: disable=invalid-name
# pylint: disable=unused-variable
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long

import unittest
import numpy as np
import grheat

class InstantaneousPoint(unittest.TestCase):

    def test_01_scalar(self):
        xp, yp, zp = 0, 0, 0.001      # meters
        point = grheat.Point(xp, yp, zp)
        tp = 0
        t = 1
        T = point.instantaneous(0, 0, 0, t, tp)

    def test_02_time_array(self):
        xp, yp, zp = 0, 0, 0.001      # meters
        point = grheat.Point(xp, yp, zp)
        tp = 0
        t = np.linspace(0,10)
        T = point.instantaneous(0, 0, 0, t, tp)

    def test_03_surface(self):
        xp, yp, zp = 0, 0, 0.001      # meters
        point = grheat.Point(xp, yp, zp)
        tp = 0
        t = 1
        X, Y = np.meshgrid(np.arange(-5, 5, 1), np.arange(-5, 5, 1))
        T = point.instantaneous(X, Y, 0, t, tp)


class ContinuousPoint(unittest.TestCase):

    def test_01_scalar(self):
        xp, yp, zp = 0, 0, 0.001      # meters
        point = grheat.Point(xp, yp, zp)
        t = 1
        T = point.continuous(0, 0, 0, t)

    def test_02_time_array(self):
        xp, yp, zp = 0, 0, 0.001      # meters
        point = grheat.Point(xp, yp, zp)
        t = np.linspace(0,10)
        T = point.continuous(0, 0, 0, t)

    def test_03_surface(self):
        xp, yp, zp = 0, 0, 0.001      # meters
        point = grheat.Point(xp, yp, zp)
        tp = 0
        t = 1
        X, Y = np.meshgrid(np.arange(-5, 5, 1), np.arange(-5, 5, 1))
        T = point.continuous(X, Y, 0, t)

    def test_04_scalar(self):
        xp, yp, zp = 0, 0, 0.001      # meters
        point = grheat.Point(xp, yp, zp)
        t = 1
        T1 = point.continuous(0, 0, 0, t)
        N= 50 
        tp = np.linspace(0,1,N)
        T = point.instantaneous(0, 0, 0, t, tp)
        T2 = T.sum()/N
        self.assertAlmostEqual(T1, T2, delta=0.01)

 
class PulsedPoint(unittest.TestCase):

    def test_01_scalar(self):
        xp, yp, zp = 0, 0, 0.001      # meters
        point = grheat.Point(xp, yp, zp)
        t_pulse = 0.5
        t = 1
        T = point.pulsed(0, 0, 0, t, t_pulse)

    def test_02_time_array(self):
        xp, yp, zp = 0, 0, 0.001      # meters
        point = grheat.Point(xp, yp, zp)
        t_pulse = 0.5
        t = np.linspace(0,10)
        T = point.pulsed(0, 0, 0, t, t_pulse)

    def test_03_surface(self):
        xp, yp, zp = 0, 0, 0.001      # meters
        point = grheat.Point(xp, yp, zp)
        t_pulse = 0.5
        t = 1
        X, Y = np.meshgrid(np.arange(-0.005, 0.005, 0.001), np.arange(-0.005, 0.005, 0.001))
        T = point.pulsed(X, Y, 0, t, t_pulse)

class InstantVsPulsed(unittest.TestCase):

    def test_01_instant(self):
        """Short pulse result should be same as instantaneous source."""
        xp, yp, zp = 0, 0, 0.001      # meters
        point = grheat.Point(xp, yp, zp)
        t_pulse = 0.00001
        tp = 0
        t = 1
        T1 = point.instantaneous(0, 0, 0, t, tp)
        T2 = point.pulsed(0, 0, 0, t, t_pulse)
        self.assertAlmostEqual(T1, T2, delta=0.001)

    def test_02_instant(self):
        """Short pulse result should be same as instantaneous source."""
        xp, yp, zp = 0, 0, 0.001      # meters
        point = grheat.Point(xp, yp, zp)
        t_pulse = 0.00001
        tp = 0
        t = np.linspace(0,10)
        T1 = point.instantaneous(0, 0, 0, t, tp)
        T2 = point.pulsed(0, 0, 0, t, t_pulse)
        self.assertAlmostEqual(T1[3], T2[3], delta=0.001)
        self.assertAlmostEqual(T1[13], T2[13], delta=0.001)

    def test_03_instant(self):
        """Short pulse result should be same as instantaneous source."""
        xp, yp, zp = 0, 0, 0.001      # meters
        point = grheat.Point(xp, yp, zp)
        t_pulse = 0.00001
        tp = 0
        t = 1
        X, Y = np.meshgrid(np.arange(-0.005, 0.005, 0.001), np.arange(-0.005, 0.005, 0.001))

        T1 = point.instantaneous(X, Y, 0, t, tp)
        T2 = point.pulsed(X, Y, 0, t, t_pulse)
        self.assertAlmostEqual(T1[0,3], T2[0,3], delta=0.001)
        self.assertAlmostEqual(T1[3,1], T2[3,1], delta=0.001)

if __name__ == '__main__':
    unittest.main()
