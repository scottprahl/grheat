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
        """
        Test if the method `instantaneous` can handle scalar input and
        compute the temperature at a single point in space at a given time.
        """
        xp, yp, zp = 0, 0, 0.001      # meters
        tp = 0
        point = grheat.Point(xp, yp, zp, tp)
        t = 1
        T = point.instantaneous(0, 0, 0, t)

    def test_02_time_array(self):
        """
        Test if the method `instantaneous` can handle an array of time values
        and compute the temperature at a single point in space across these times.
        """
        xp, yp, zp = 0, 0, 0.001      # meters
        tp = 0
        point = grheat.Point(xp, yp, zp, tp)
        t = np.linspace(0, 10)
        T = point.instantaneous(0, 0, 0, t)

    def test_03_surface(self):
        """
        Test if the method `instantaneous` can handle meshgrid input for x and y
        coordinates and compute the temperature across a surface at a given time.
        """
        xp, yp, zp = 0, 0, 0.001      # meters
        tp = 0
        point = grheat.Point(xp, yp, zp, tp)
        t = 1
        X, Y = np.meshgrid(np.arange(-5, 5, 1), np.arange(-5, 5, 1))
        T = point.instantaneous(X, Y, 0, t)

    def test_invalid_boundary_01(self):
        """
        Test if passing an invalid boundary value 'bad_value' raises a ValueError.
        """
        with self.assertRaises(ValueError):
            grheat.Point(0, 0, 0.001, boundary='bad_value')

    def test_04_surface(self):
        """
        Verify that we still get an array zeros if t=tp.
        """
        tp = 0                          # seconds
        t = tp                          # seconds
        xp, yp, zp = 0, 0.0001, 0.001   # m
        x = 0                           # m
        y = 0                           # m
        z = np.linspace(0, 0.002, 101)  # m
        zz = 1000 * z                   # mm

        point = grheat.Point(xp, yp, zp, tp, boundary='zero')
        T = point.instantaneous(x, y, z, t)
        self.assertIsInstance(T, np.ndarray)     # is an array
        self.assertTrue(np.all(np.equal(T, 0)))  # are all zeros
        self.assertEqual(T.shape, z.shape)       # same shape

    def test_05_surface(self):
        """
        Verify that we still get an array zeros if t<tp.
        """
        tp = 1                          # seconds
        t = 0.5                         # seconds
        xp, yp, zp = 0, 0.0001, 0.001   # m
        x = 0                           # m
        y = 0                           # m
        z = np.linspace(0, 0.002, 101)  # m
        zz = 1000 * z                   # mm

        point = grheat.Point(xp, yp, zp, tp, boundary='zero')
        T = point.instantaneous(x, y, z, t)
        self.assertIsInstance(T, np.ndarray)     # is an array
        self.assertTrue(np.all(np.equal(T, 0)))  # are all zeros
        self.assertEqual(T.shape, z.shape)       # same shape

    def test_06_surface(self):
        """
        Verify that we still get an array zeros if t<tp.
        """
        tp = 1                          # seconds
        t = 0.5                         # seconds
        xp, yp, zp = 0, 0.0001, 0.001   # m
        x = 0                           # m
        y = 0                           # m
        z = 0.001                       # m
        zz = 1000 * z                   # mm

        point = grheat.Point(xp, yp, zp, tp)
        T = point.instantaneous(x, y, z, t)
        self.assertEqual(T, 0)


class ContinuousPoint(unittest.TestCase):

    def test_01_scalar(self):
        """
        Test if the method `continuous` can handle scalar input and
        compute the temperature at a single point in space at a given time.
        """
        xp, yp, zp = 0, 0, 0.001      # meters
        point = grheat.Point(xp, yp, zp)
        t = 1
        T = point.continuous(0, 0, 0, t)

    def test_02_time_array(self):
        """
        Test if the method `continuous` can handle an array of time values
        and compute the temperature at a single point in space across these times.
        """
        xp, yp, zp = 0, 0, 0.001      # meters
        point = grheat.Point(xp, yp, zp)
        t = np.linspace(0, 10)
        T = point.continuous(0, 0, 0, t)

    def test_03_surface(self):
        """
        Test if the method `continuous` can handle meshgrid input for x and y
        coordinates and compute the temperature across a surface at a given time.
        """
        xp, yp, zp = 0, 0, 0.001      # meters
        point = grheat.Point(xp, yp, zp)
        tp = 0
        t = 1
        X, Y = np.meshgrid(np.arange(-5, 5, 1), np.arange(-5, 5, 1))
        T = point.continuous(X, Y, 0, t)

    def test_04_scalar(self):
        """
        Test if the method `continuous`'s output for a scalar input is
        consistent with the averaged output of method `instantaneous`
        over a range of time points.
        """
        N = 50
        xp, yp, zp = 0, 0, 0.001      # meters
        tp = np.linspace(0, 1, N)
        point = grheat.Point(xp, yp, zp, tp)
        t = 1
        T1 = point.continuous(0, 0, 0, t)
        T = point.instantaneous(0, 0, 0, t)
        T2 = T.sum() / N
        self.assertAlmostEqual(T1, T2, delta=0.01)

    def test_05_surface(self):
        """
        Verify that we still get an array zeros if t=tp.
        """
        t = -1                          # seconds
        xp, yp, zp = 0, 0.0001, 0.001   # m
        x = 0                           # m
        y = 0                           # m
        z = np.linspace(0, 0.002, 101)  # m
        zz = 1000 * z                   # mm

        point = grheat.Point(xp, yp, zp, boundary='zero')
        T = point.continuous(x, y, z, t)
        self.assertIsInstance(T, np.ndarray)     # is an array
        self.assertTrue(np.all(np.equal(T, 0)))  # are all zeros
        self.assertEqual(T.shape, z.shape)       # same shape

    def test_06_surface(self):
        """
        Verify that we still get an array zeros if t<tp.
        """
        t = 0                           # seconds
        xp, yp, zp = 0, 0.0001, 0.001   # m
        x = 0                           # m
        y = 0                           # m
        z = np.linspace(0, 0.002, 101)  # m
        zz = 1000 * z                   # mm

        point = grheat.Point(xp, yp, zp, boundary='zero')
        T = point.continuous(x, y, z, t)
        self.assertIsInstance(T, np.ndarray)     # is an array
        self.assertTrue(np.all(np.equal(T, 0)))  # are all zeros
        self.assertEqual(T.shape, z.shape)       # same shape

    def test_07_surface(self):
        """
        Verify that we still get an array zeros if t<tp.
        """
        t = 0                           # seconds
        xp, yp, zp = 0, 0.0001, 0.001   # m
        x = 0                           # m
        y = 0                           # m
        z = 0.001                       # m
        zz = 1000 * z                   # mm

        point = grheat.Point(xp, yp, zp, boundary='zero')
        T = point.continuous(x, y, z, t)
        self.assertEqual(T, 0)


class PulsedPoint(unittest.TestCase):

    def test_01_scalar(self):
        """
        Test if the method `pulsed` can handle scalar input and
        compute the temperature at a single point in space at a given time,
        following a pulse at a specified earlier time.
        """
        xp, yp, zp = 0, 0, 0.001      # meters
        point = grheat.Point(xp, yp, zp)
        t_pulse = 0.5
        t = 1
        T = point.pulsed(0, 0, 0, t, t_pulse)

    def test_02_time_array(self):
        """
        Test if the method `pulsed` can handle an array of time values
        and compute the temperature at a single point in space across these times,
        following a pulse at a specified earlier time.
        """
        xp, yp, zp = 0, 0, 0.001      # meters
        point = grheat.Point(xp, yp, zp)
        t_pulse = 0.5
        t = np.linspace(0, 10)
        T = point.pulsed(0, 0, 0, t, t_pulse)

    def test_03_surface(self):
        """
        Test if the method `pulsed` can handle meshgrid input for x and y
        coordinates and compute the temperature across a surface at a given time,
        following a pulse at a specified earlier time.
        """
        xp, yp, zp = 0, 0, 0.001      # meters
        point = grheat.Point(xp, yp, zp)
        t_pulse = 0.5
        t = 1
        X, Y = np.meshgrid(np.arange(-0.005, 0.005, 0.001), np.arange(-0.005, 0.005, 0.001))
        T = point.pulsed(X, Y, 0, t, t_pulse)


class InstantVsPulsed(unittest.TestCase):

    def test_01_instant(self):
        """Short pulse result should be same as instantaneous source for scalars."""
        xp, yp, zp = 0, 0, 0.001      # meters
        point = grheat.Point(xp, yp, zp)
        t_pulse = 0.00001
        t = 1
        T1 = point.instantaneous(0, 0, 0, t)
        T2 = point.pulsed(0, 0, 0, t, t_pulse)
        self.assertAlmostEqual(T1, T2, delta=0.001)

    def test_02_instant(self):
        """Short pulse result should be same as instantaneous source for arrays."""
        xp, yp, zp = 0, 0, 0.001      # meters
        point = grheat.Point(xp, yp, zp)
        t_pulse = 0.00001
        t = np.linspace(0, 10)
        T1 = point.instantaneous(0, 0, 0, t)
        T2 = point.pulsed(0, 0, 0, t, t_pulse)
        self.assertAlmostEqual(T1[3], T2[3], delta=0.001)
        self.assertAlmostEqual(T1[13], T2[13], delta=0.001)

    def test_03_instant(self):
        """Short pulse result should be same as instantaneous source for mesh."""
        xp, yp, zp = 0, 0, 0.001      # meters
        point = grheat.Point(xp, yp, zp)
        t_pulse = 0.00001
        t = 1
        X, Y = np.meshgrid(np.arange(-0.005, 0.005, 0.001), np.arange(-0.005, 0.005, 0.001))

        T1 = point.instantaneous(X, Y, 0, t)
        T2 = point.pulsed(X, Y, 0, t, t_pulse)
        self.assertAlmostEqual(T1[0, 3], T2[0, 3], delta=0.001)
        self.assertAlmostEqual(T1[3, 1], T2[3, 1], delta=0.001)


class ConstantBoundary(unittest.TestCase):

    def test_01_zero(self):
        """Surface temperature should be zero."""
        xp, yp, zp = 0, 0, 0.001      # meters
        point = grheat.Point(xp, yp, zp, boundary='zero')
        t_pulse = 1
        t = 2
        T = point.pulsed(0, 0, 0, t, t_pulse)
        self.assertEqual(T, 0)

    def test_02_zero(self):
        """Surface temperature should be zero at all times."""
        xp, yp, zp = 0, 0, 0.001      # meters
        point = grheat.Point(xp, yp, zp, boundary='zero')
        t_pulse = 1
        t = np.linspace(0, 10)
        T = point.pulsed(0, 0, 0, t, t_pulse)
        self.assertEqual(T[3], 0)
        self.assertEqual(T[13], 0)

    def test_03_zero(self):
        """Surface temperature should be zero at all locations."""
        xp, yp, zp = 0, 0, 0.0001      # meters
        point = grheat.Point(xp, yp, zp, boundary='zero')
        t_pulse = 1
        t = 1.1
        X, Y = np.meshgrid(np.arange(-0.0005, 0.0005, 0.0001), np.arange(-0.0005, 0.0005, 0.0001))
        T = point.pulsed(X, Y, 0, t, t_pulse)
        self.assertEqual(T[0, 3], 0)
        self.assertEqual(T[3, 1], 0)


class AdiabaticBoundary(unittest.TestCase):

    def test_01_adiabatic(self):
        """Temperature should be equal above and below."""
        xp, yp, zp = 0, 0, 0.001      # meters
        point = grheat.Point(xp, yp, zp, boundary='adiabatic')
        t_pulse = 1
        t = 2
        T1 = point.pulsed(0, 0, +0.0001, t, t_pulse)
        T2 = point.pulsed(0, 0, -0.0001, t, t_pulse)
        self.assertAlmostEqual(T1, T2, delta=1e-8)

    def test_02_adiabatic(self):
        """Temperature should be equal above and below at all times."""
        xp, yp, zp = 0, 0, 0.001      # meters
        point = grheat.Point(xp, yp, zp, boundary='adiabatic')
        t_pulse = 1
        t = np.linspace(0, 2)
        T1 = point.pulsed(0, 0, +0.0001, t, t_pulse)
        T2 = point.pulsed(0, 0, -0.0001, t, t_pulse)
        self.assertAlmostEqual(T1[3], T2[3], delta=1e-8)
        self.assertAlmostEqual(T1[13], T2[13], delta=1e-8)

    def test_03_adiabatic(self):
        """Temperature should be equal above and below."""
        xp, yp, zp = 0, 0, 0.001      # meters
        point = grheat.Point(xp, yp, zp, boundary='adiabatic')
        t_pulse = 1
        t = 1.1
        X, Y = np.meshgrid(np.arange(-0.0005, 0.0005, 0.0001), np.arange(-0.0005, 0.0005, 0.0001))
        T1 = point.pulsed(X, Y, +0.0001, t, t_pulse)
        T2 = point.pulsed(X, Y, -0.0001, t, t_pulse)
        self.assertAlmostEqual(T1[0, 3], T2[0, 3], delta=1e-8)
        self.assertAlmostEqual(T1[3, 1], T2[3, 1], delta=1e-8)


if __name__ == '__main__':
    unittest.main()
