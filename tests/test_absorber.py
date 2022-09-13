#! /usr/bin/env python3
# pylint: disable=invalid-name
# pylint: disable=unused-variable
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-module-docstring

import unittest
import numpy as np
import grheat

class InstantaneousAbsorber(unittest.TestCase):

    def test_01_scalar(self):
        mua = 1000  # 1/meter
        tp = 0
        t = 1
        medium = grheat.Absorber(mua)
        T = medium.instantaneous(0, t, tp)

    def test_02_time_array(self):
        mua = 1000  # 1/meter
        tp = 0
        t = np.linspace(0,10)
        medium = grheat.Absorber(mua)
        T = medium.instantaneous(0, t, tp)


class ContinuousAbsorber(unittest.TestCase):

    def test_01_scalar(self):
        mua = 1000  # 1/meter
        t = 1
        medium = grheat.Absorber(mua)
        T = medium.continuous(0, t)

    def test_02_time_array(self):
        mua = 1000  # 1/meter
        t = np.linspace(0,10)
        medium = grheat.Absorber(mua)
        T = medium.continuous(0, t)


class PulsedAbsorber(unittest.TestCase):

    def test_01_scalar(self):
        mua = 1000  # 1/meter
        t_pulse = 0.5
        t = 1
        medium = grheat.Absorber(mua)
        T = medium.pulsed(0, t, t_pulse)

    def test_02_time_array(self):
        mua = 1000  # 1/meter
        t_pulse = 0.5
        t = np.linspace(0,10)
        medium = grheat.Absorber(mua)
        T = medium.pulsed(0, t, t_pulse)


class InstantVsPulsed(unittest.TestCase):

    def test_01_instant(self):
        """Short pulse result should be same as instantaneous source."""
        mua = 100  # 1/meter
        t_pulse = 0.00001
        tp = 0
        t = 2
        z = 0.001
        radiant_exposure = 1e6    # 1 J/mm² in J/m²
        medium = grheat.Absorber(mua)
        T1 = radiant_exposure * medium.instantaneous(z, t, tp)
        T2 = radiant_exposure * medium.pulsed(z, t, t_pulse)
        print(T1/T2)
        print(T2/T1)
        self.assertAlmostEqual(T1, T2, delta=0.001)

#     def test_02_instant(self):
#         """Short pulse result should be same as instantaneous source."""
#         mua = 1000  # 1/meter
#         t_pulse = 0.001
#         tp = 0
#         t = np.linspace(0,10)
#         radiant_exposure = 1e6    # 1 J/mm² in J/m²
#         medium = grheat.Absorber(mua)
#         T1 = radiant_exposure * medium.instantaneous(0, t, tp)
#         T2 = radiant_exposure * medium.pulsed(0, t, t_pulse)
#         self.assertAlmostEqual(T1[3], T2[3], delta=0.001)
#         self.assertAlmostEqual(T1[13], T2[13], delta=0.001)

class ConstantBoundary(unittest.TestCase):

    def test_01_zero(self):
        """Surface temperature should be zero."""
        mua = 1000  # 1/meter
        medium = grheat.Absorber(mua, boundary='zero')
        t_pulse = 1
        t = 2
        T = medium.pulsed(0, t, t_pulse)
        self.assertEqual(T, 0)

    def test_02_zero(self):
        """Surface temperature should be zero at all times."""
        mua = 1000  # 1/meter
        medium = grheat.Absorber(mua, boundary='zero')
        t_pulse = 1
        t = np.linspace(0,10)
        T = medium.pulsed(0, t, t_pulse)
        self.assertEqual(T[3], 0)
        self.assertEqual(T[13], 0)


class AdiabaticBoundary(unittest.TestCase):

    def test_01_adiabatic(self):
        """Temperature should be equal above and below."""
        mua = 1000  # 1/meter
        medium = grheat.Absorber(mua, boundary='adiabatic')
        t_pulse = 1
        t = 2
        T1 = medium.pulsed(+0.0001, t, t_pulse)
        T2 = medium.pulsed(-0.0001, t, t_pulse)
        self.assertAlmostEqual(T1, T2, delta=1e-8)

    def test_02_adiabatic(self):
        """Temperature should be equal above and below at all times."""
        mua = 1000  # 1/meter
        medium = grheat.Absorber(mua, boundary='adiabatic')
        t_pulse = 1
        t = np.linspace(0,2)
        T1 = medium.pulsed(+0.0001, t, t_pulse)
        T2 = medium.pulsed(-0.0001, t, t_pulse)
        self.assertAlmostEqual(T1[3], T2[3], delta=1e-8)
        self.assertAlmostEqual(T1[13], T2[13], delta=1e-8)


if __name__ == '__main__':
    unittest.main()
