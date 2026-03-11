#! /usr/bin/env python3
"""Regression tests for absorbing-point Green's function solutions."""

import numpy as np
import pytest

import grheat


def make_source():
    """Create a reusable absorbing-point source for tests."""
    return grheat.AbsorbingPoint(mu_a=1 * 1000, xp=0, yp=0, boundary="infinite", n_quad=100)


class TestAbsorbingPointInstant:
    """Check instantaneous absorbing-point solutions."""

    def test_instantaneous_scalar(self):
        """Verify scalar instantaneous temperatures for an absorbing point source."""
        x, y, z, t = 0.0001, 0, 0, 1
        expected_temperature = 3.343552
        temperature = make_source().instantaneous(x, y, z, t) * 1e6
        assert temperature == pytest.approx(expected_temperature, abs=1e-5)

    def test_instantaneous_array_time(self):
        """Verify vectorized instantaneous temperatures for an absorbing point source."""
        x, y, z = 0.0001, 0, 0
        t = np.array([1, 2, 3, 4, 5])
        expected_temperature = np.array([3.343552, 1.843356, 1.176335, 0.830986, 0.626676])
        temperature = make_source().instantaneous(x, y, z, t) * 1e6
        np.testing.assert_allclose(temperature, expected_temperature, rtol=1e-5)


class TestAbsorbingPointContinuous:
    """Check continuous absorbing-point solutions."""

    def test_continuous_scalar(self):
        """Verify scalar continuous temperatures for an absorbing point source."""
        x, y, z, t = 0.0001, 0, 0, 3
        expected_temperature = 6.856171
        temperature = make_source().continuous(x, y, z, t) * 1e6
        assert temperature == pytest.approx(expected_temperature, abs=1e-5)

    def test_continuous_array_time(self):
        """Verify vectorized continuous temperatures for an absorbing point source."""
        x, y, z = 0.0001, 0, 0
        t = np.array([1, 2, 3, 4, 5])
        expected_temperature = np.array([2.895831, 5.386845, 6.856171, 7.843278, 8.564113])
        temperature = make_source().continuous(x, y, z, t) * 1e6
        np.testing.assert_allclose(temperature, expected_temperature, rtol=1e-5)


class TestAbsorbingPointPulsed:
    """Check pulsed absorbing-point solutions."""

    def test_pulsed_scalar(self):
        """Verify scalar pulsed temperatures for an absorbing point source."""
        x, y, z, t, t_pulse = 0.0001, 0, 0, 1, 0.5
        expected_temperature = 3.816487
        temperature = make_source().pulsed(x, y, z, t, t_pulse) * 1e6
        assert temperature == pytest.approx(expected_temperature, abs=1e-5)

    def test_pulsed_array_time(self):
        """Verify vectorized pulsed temperatures for an absorbing point source."""
        x, y, z, t_pulse = 0.0001, 0, 0, 0.5
        t = np.array([1, 2, 3, 4, 5])
        expected_temperature = np.array([3.816487, 2.119566, 1.305044, 0.901593, 0.670086])
        temperature = make_source().pulsed(x, y, z, t, t_pulse) * 1e6
        np.testing.assert_allclose(temperature, expected_temperature, rtol=1e-5)
