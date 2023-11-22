#! /usr/bin/env python3
# pylint: disable=invalid-name
# pylint: disable=unused-variable
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-module-docstring

import unittest
import numpy as np
import grheat
joules_per_calorie = 4.184


class TestAbsorbingPointInstant(unittest.TestCase):

    def setUp(self):
        # Create an AbsorbingPoints instance to use in tests
        self.source = grheat.AbsorbingPoint(
            mu_a=1 * 1000,
            xp=0, yp=0,
            boundary='infinite',
            n_quad=100
        )

    def test_instantaneous_scalar(self):
        x, y, z, t = 0.0001, 0, 0, 1
        expected_temperature = 3.343552
        temperature = self.source.instantaneous(x, y, z, t) * 1e6
        self.assertAlmostEqual(temperature, expected_temperature, places=5)

    def test_instantaneous_array_time(self):
        x, y, z = 0.0001, 0, 0
        t = np.array([1, 2, 3, 4, 5])
        # Assume an expected temperature array for these parameters
        expected_temperature = np.array([3.343552, 1.843356, 1.176335, 0.830986, 0.626676])
        temperature = self.source.instantaneous(x, y, z, t) * 1e6
        np.testing.assert_allclose(temperature, expected_temperature, rtol=1e-5)


class TestAbsorbingPointContinuous(unittest.TestCase):

    def setUp(self):
        # Create an AbsorbingPoints instance to use in tests
        self.source = grheat.AbsorbingPoint(
            mu_a=1 * 1000,
            xp=0, yp=0,
            boundary='infinite',
            n_quad=100
        )

    def test_continuous_scalar(self):
        x, y, z, t = 0.0001, 0, 0, 3
        expected_temperature = 2.895831
        temperature = self.source.continuous(x, y, z, t) * 1e6
        self.assertAlmostEqual(temperature, expected_temperature, places=5)

    def test_continuous_array_time(self):
        x, y, z = 0.0001, 0, 0
        t = np.array([1, 2, 3, 4, 5])
        # Assume an expected temperature array for these parameters
        expected_temperature = np.array([0.02, 0.04, 0.06, 0.08, 0.10])
        temperature = self.source.continuous(x, y, z, t) * 1e6
        np.testing.assert_allclose(temperature, expected_temperature, rtol=1e-5)


class TestAbsorbingPointPulsed(unittest.TestCase):

    def setUp(self):
        # Create an AbsorbingPoints instance to use in tests
        self.source = grheat.AbsorbingPoint(
            mu_a=1 * 1000,
            xp=0, yp=0,
            boundary='infinite',
            n_quad=100
        )

    def test_pulsed_scalar(self):
        x, y, z, t, t_pulse = 0.0001, 0, 0, 1, 0.5
        expected_temperature = 3.816487
        temperature = self.source.pulsed(x, y, z, t, t_pulse) * 1e6
        self.assertAlmostEqual(temperature, expected_temperature, places=5)

    def test_pulsed_array_time(self):
        x, y, z, t_pulse = 0.0001, 0, 0, 0.5
        t = np.array([1, 2, 3, 4, 5])
        # Assume an expected temperature array for these parameters
        expected_temperature = np.array([3.816487, 2.119566, 1.305044, 0.901593, 0.670086])
        temperature = self.source.pulsed(x, y, z, t, t_pulse) * 1e6
        np.testing.assert_allclose(temperature, expected_temperature, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
