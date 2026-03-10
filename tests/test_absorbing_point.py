#! /usr/bin/env python3
# pylint: disable=invalid-name
# pylint: disable=unused-variable
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-module-docstring

import numpy as np
import pytest

import grheat


class TestAbsorbingPointInstant:
    def setup_method(self):
        """Create a reusable absorbing-point source for instantaneous tests."""
        self.source = grheat.AbsorbingPoint(mu_a=1 * 1000, xp=0, yp=0, boundary="infinite", n_quad=100)

    def test_instantaneous_scalar(self):
        """Verify scalar instantaneous temperatures for an absorbing point source."""
        x, y, z, t = 0.0001, 0, 0, 1
        expected_temperature = 3.343552
        temperature = self.source.instantaneous(x, y, z, t) * 1e6
        assert temperature == pytest.approx(expected_temperature, abs=1e-5)

    def test_instantaneous_array_time(self):
        """Verify vectorized instantaneous temperatures for an absorbing point source."""
        x, y, z = 0.0001, 0, 0
        t = np.array([1, 2, 3, 4, 5])
        expected_temperature = np.array([3.343552, 1.843356, 1.176335, 0.830986, 0.626676])
        temperature = self.source.instantaneous(x, y, z, t) * 1e6
        np.testing.assert_allclose(temperature, expected_temperature, rtol=1e-5)


class TestAbsorbingPointContinuous:
    def setup_method(self):
        """Create a reusable absorbing-point source for continuous tests."""
        self.source = grheat.AbsorbingPoint(mu_a=1 * 1000, xp=0, yp=0, boundary="infinite", n_quad=100)

    def test_continuous_scalar(self):
        """Verify scalar continuous temperatures for an absorbing point source."""
        x, y, z, t = 0.0001, 0, 0, 3
        expected_temperature = 6.856171
        temperature = self.source.continuous(x, y, z, t) * 1e6
        assert temperature == pytest.approx(expected_temperature, abs=1e-5)

    def test_continuous_array_time(self):
        """Verify vectorized continuous temperatures for an absorbing point source."""
        x, y, z = 0.0001, 0, 0
        t = np.array([1, 2, 3, 4, 5])
        expected_temperature = np.array([2.895831, 5.386845, 6.856171, 7.843278, 8.564113])
        temperature = self.source.continuous(x, y, z, t) * 1e6
        np.testing.assert_allclose(temperature, expected_temperature, rtol=1e-5)


class TestAbsorbingPointPulsed:
    def setup_method(self):
        """Create a reusable absorbing-point source for pulsed tests."""
        self.source = grheat.AbsorbingPoint(mu_a=1 * 1000, xp=0, yp=0, boundary="infinite", n_quad=100)

    def test_pulsed_scalar(self):
        """Verify scalar pulsed temperatures for an absorbing point source."""
        x, y, z, t, t_pulse = 0.0001, 0, 0, 1, 0.5
        expected_temperature = 3.816487
        temperature = self.source.pulsed(x, y, z, t, t_pulse) * 1e6
        assert temperature == pytest.approx(expected_temperature, abs=1e-5)

    def test_pulsed_array_time(self):
        """Verify vectorized pulsed temperatures for an absorbing point source."""
        x, y, z, t_pulse = 0.0001, 0, 0, 0.5
        t = np.array([1, 2, 3, 4, 5])
        expected_temperature = np.array([3.816487, 2.119566, 1.305044, 0.901593, 0.670086])
        temperature = self.source.pulsed(x, y, z, t, t_pulse) * 1e6
        np.testing.assert_allclose(temperature, expected_temperature, rtol=1e-5)
