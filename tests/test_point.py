#! /usr/bin/env python3
"""Regression tests for point-source Green's function solutions."""

import numpy as np
import pytest

import grheat


class TestInstantaneousPoint:
    """Check instantaneous point-source solutions."""

    def test_01_scalar(self):
        """
        Test if the method `instantaneous` can handle scalar input and
        compute the temperature at a single point in space at a given time.
        """
        xp, yp, zp = 0, 0, 0.001  # meters
        tp = 0
        point = grheat.Point(xp, yp, zp, tp)
        t = 1
        point.instantaneous(0, 0, 0, t)

    def test_02_time_array(self):
        """
        Test if the method `instantaneous` can handle an array of time values
        and compute the temperature at a single point in space across these times.
        """
        xp, yp, zp = 0, 0, 0.001  # meters
        tp = 0
        point = grheat.Point(xp, yp, zp, tp)
        t = np.linspace(0, 10)
        point.instantaneous(0, 0, 0, t)

    def test_03_surface(self):
        """
        Test if the method `instantaneous` can handle meshgrid input for x and y
        coordinates and compute the temperature across a surface at a given time.
        """
        xp, yp, zp = 0, 0, 0.001  # meters
        tp = 0
        point = grheat.Point(xp, yp, zp, tp)
        t = 1
        xx, yy = np.meshgrid(np.arange(-5, 5, 1), np.arange(-5, 5, 1))
        point.instantaneous(xx, yy, 0, t)

    def test_invalid_boundary_01(self):
        """
        Test if passing an invalid boundary value 'bad_value' raises a ValueError.
        """
        with pytest.raises(ValueError):
            grheat.Point(0, 0, 0.001, boundary="bad_value")

    def test_04_surface(self):
        """
        Verify that we still get an array zeros if t=tp.
        """
        tp = 0  # seconds
        t = tp  # seconds
        xp, yp, zp = 0, 0.0001, 0.001  # m
        x = 0  # m
        y = 0  # m
        z = np.linspace(0, 0.002, 101)  # m

        point = grheat.Point(xp, yp, zp, tp, boundary="zero")
        temperature = point.instantaneous(x, y, z, t)
        assert isinstance(temperature, np.ndarray)
        assert np.all(np.equal(temperature, 0))
        assert temperature.shape == z.shape

    def test_05_surface(self):
        """
        Verify that we still get an array zeros if t<tp.
        """
        tp = 1  # seconds
        t = 0.5  # seconds
        xp, yp, zp = 0, 0.0001, 0.001  # m
        x = 0  # m
        y = 0  # m
        z = np.linspace(0, 0.002, 101)  # m

        point = grheat.Point(xp, yp, zp, tp, boundary="zero")
        temperature = point.instantaneous(x, y, z, t)
        assert isinstance(temperature, np.ndarray)
        assert np.all(np.equal(temperature, 0))
        assert temperature.shape == z.shape

    def test_06_surface(self):
        """
        Verify that we still get an array zeros if t<tp.
        """
        tp = 1  # seconds
        t = 0.5  # seconds
        xp, yp, zp = 0, 0.0001, 0.001  # m
        x = 0  # m
        y = 0  # m
        z = 0.001  # m

        point = grheat.Point(xp, yp, zp, tp)
        temperature = point.instantaneous(x, y, z, t)
        assert temperature == 0


class TestContinuousPoint:
    """Check continuous point-source solutions."""

    def test_01_scalar(self):
        """
        Test if the method `continuous` can handle scalar input and
        compute the temperature at a single point in space at a given time.
        """
        xp, yp, zp = 0, 0, 0.001  # meters
        point = grheat.Point(xp, yp, zp)
        t = 1
        point.continuous(0, 0, 0, t)

    def test_02_time_array(self):
        """
        Test if the method `continuous` can handle an array of time values
        and compute the temperature at a single point in space across these times.
        """
        xp, yp, zp = 0, 0, 0.001  # meters
        point = grheat.Point(xp, yp, zp)
        t = np.linspace(0, 10)
        point.continuous(0, 0, 0, t)

    def test_03_surface(self):
        """
        Test if the method `continuous` can handle meshgrid input for x and y
        coordinates and compute the temperature across a surface at a given time.
        """
        xp, yp, zp = 0, 0, 0.001  # meters
        point = grheat.Point(xp, yp, zp)
        t = 1
        xx, yy = np.meshgrid(np.arange(-5, 5, 1), np.arange(-5, 5, 1))
        point.continuous(xx, yy, 0, t)

    def test_04_scalar(self):
        """
        Test if the method `continuous`'s output for a scalar input is
        consistent with the averaged output of method `instantaneous`
        over a range of time points.
        """
        n_pulses = 50
        xp, yp, zp = 0, 0, 0.001  # meters
        t = 1
        point = grheat.Point(xp, yp, zp)
        temperature1 = point.continuous(0, 0, 0, t)

        point.tp = np.linspace(0, t, n_pulses)
        temperature = point.instantaneous(0, 0, 0, t)
        temperature2 = temperature.sum() / n_pulses
        assert temperature1 == pytest.approx(temperature2, abs=0.01)

    def test_05_surface(self):
        """
        Verify that we still get an array zeros if t=tp.
        """
        t = -1  # seconds
        xp, yp, zp = 0, 0.0001, 0.001  # m
        x = 0  # m
        y = 0  # m
        z = np.linspace(0, 0.002, 101)  # m

        point = grheat.Point(xp, yp, zp, boundary="zero")
        temperature = point.continuous(x, y, z, t)
        assert isinstance(temperature, np.ndarray)
        assert np.all(np.equal(temperature, 0))
        assert temperature.shape == z.shape

    def test_06_surface(self):
        """
        Verify that we still get an array zeros if t<tp.
        """
        t = 0  # seconds
        xp, yp, zp = 0, 0.0001, 0.001  # m
        x = 0  # m
        y = 0  # m
        z = np.linspace(0, 0.002, 101)  # m

        point = grheat.Point(xp, yp, zp, boundary="zero")
        temperature = point.continuous(x, y, z, t)
        assert isinstance(temperature, np.ndarray)
        assert np.all(np.equal(temperature, 0))
        assert temperature.shape == z.shape

    def test_07_surface(self):
        """
        Verify that we still get an array zeros if t<tp.
        """
        t = 0  # seconds
        xp, yp, zp = 0, 0.0001, 0.001  # m
        x = 0  # m
        y = 0  # m
        z = 0.001  # m

        point = grheat.Point(xp, yp, zp, boundary="zero")
        temperature = point.continuous(x, y, z, t)
        assert temperature == 0


class TestPulsedPoint:
    """Check pulsed point-source solutions."""

    def test_01_scalar(self):
        """
        Test if the method `pulsed` can handle scalar input and
        compute the temperature at a single point in space at a given time,
        following a pulse at a specified earlier time.
        """
        xp, yp, zp = 0, 0, 0.001  # meters
        point = grheat.Point(xp, yp, zp)
        t_pulse = 0.5
        t = 1
        point.pulsed(0, 0, 0, t, t_pulse)

    def test_02_time_array(self):
        """
        Test if the method `pulsed` can handle an array of time values
        and compute the temperature at a single point in space across these times,
        following a pulse at a specified earlier time.
        """
        xp, yp, zp = 0, 0, 0.001  # meters
        point = grheat.Point(xp, yp, zp)
        t_pulse = 0.5
        t = np.linspace(0, 10)
        point.pulsed(0, 0, 0, t, t_pulse)

    def test_03_surface(self):
        """
        Test if the method `pulsed` can handle meshgrid input for x and y
        coordinates and compute the temperature across a surface at a given time,
        following a pulse at a specified earlier time.
        """
        xp, yp, zp = 0, 0, 0.001  # meters
        point = grheat.Point(xp, yp, zp)
        t_pulse = 0.5
        t = 1
        xx, yy = np.meshgrid(np.arange(-0.005, 0.005, 0.001), np.arange(-0.005, 0.005, 0.001))
        point.pulsed(xx, yy, 0, t, t_pulse)


class TestInstantVsPulsed:
    """Compare short pulses against instantaneous point-source results."""

    def test_01_instant(self):
        """Short pulse result should be same as instantaneous source for scalars."""
        xp, yp, zp = 0, 0, 0.001  # meters
        point = grheat.Point(xp, yp, zp)
        t_pulse = 0.00001
        t = 1
        t1 = point.instantaneous(0, 0, 0, t)
        t2 = point.pulsed(0, 0, 0, t, t_pulse)
        assert t1 == pytest.approx(t2, abs=0.001)

    def test_02_instant(self):
        """Short pulse result should be same as instantaneous source for arrays."""
        xp, yp, zp = 0, 0, 0.001  # meters
        point = grheat.Point(xp, yp, zp)
        t_pulse = 0.00001
        t = np.linspace(0, 10)
        t1 = point.instantaneous(0, 0, 0, t)
        t2 = point.pulsed(0, 0, 0, t, t_pulse)
        assert t1[3] == pytest.approx(t2[3], abs=0.001)
        assert t1[13] == pytest.approx(t2[13], abs=0.001)

    def test_03_instant(self):
        """Short pulse result should be same as instantaneous source for mesh."""
        xp, yp, zp = 0, 0, 0.001  # meters
        point = grheat.Point(xp, yp, zp)
        t_pulse = 0.00001
        t = 1
        xx, yy = np.meshgrid(np.arange(-0.005, 0.005, 0.001), np.arange(-0.005, 0.005, 0.001))

        t1 = point.instantaneous(xx, yy, 0, t)
        t2 = point.pulsed(xx, yy, 0, t, t_pulse)
        assert t1[0, 3] == pytest.approx(t2[0, 3], abs=0.001)
        assert t1[3, 1] == pytest.approx(t2[3, 1], abs=0.001)


class TestConstantBoundary:
    """Check zero-temperature boundary behavior for point sources."""

    def test_01_zero(self):
        """Surface temperature should be zero."""
        xp, yp, zp = 0, 0, 0.001  # meters
        point = grheat.Point(xp, yp, zp, boundary="zero")
        t_pulse = 1
        t = 2
        temperature = point.pulsed(0, 0, 0, t, t_pulse)
        assert temperature == 0

    def test_02_zero(self):
        """Surface temperature should be zero at all times."""
        xp, yp, zp = 0, 0, 0.001  # meters
        point = grheat.Point(xp, yp, zp, boundary="zero")
        t_pulse = 1
        t = np.linspace(0, 10)
        temperature = point.pulsed(0, 0, 0, t, t_pulse)
        assert temperature[3] == 0
        assert temperature[13] == 0

    def test_03_zero(self):
        """Surface temperature should be zero at all locations."""
        xp, yp, zp = 0, 0, 0.0001  # meters
        point = grheat.Point(xp, yp, zp, boundary="zero")
        t_pulse = 1
        t = 1.1
        xx, yy = np.meshgrid(np.arange(-0.0005, 0.0005, 0.0001), np.arange(-0.0005, 0.0005, 0.0001))
        temperature = point.pulsed(xx, yy, 0, t, t_pulse)
        assert temperature[0, 3] == 0
        assert temperature[3, 1] == 0


class TestAdiabaticBoundary:
    """Check adiabatic boundary symmetry for point sources."""

    def test_01_adiabatic(self):
        """Temperature should be equal above and below."""
        xp, yp, zp = 0, 0, 0.001  # meters
        point = grheat.Point(xp, yp, zp, boundary="adiabatic")
        t_pulse = 1
        t = 2
        t1 = point.pulsed(0, 0, +0.0001, t, t_pulse)
        t2 = point.pulsed(0, 0, -0.0001, t, t_pulse)
        assert t1 == pytest.approx(t2, abs=1e-8)

    def test_02_adiabatic(self):
        """Temperature should be equal above and below at all times."""
        xp, yp, zp = 0, 0, 0.001  # meters
        point = grheat.Point(xp, yp, zp, boundary="adiabatic")
        t_pulse = 1
        t = np.linspace(0, 2)
        t1 = point.pulsed(0, 0, +0.0001, t, t_pulse)
        t2 = point.pulsed(0, 0, -0.0001, t, t_pulse)
        assert t1[3] == pytest.approx(t2[3], abs=1e-8)
        assert t1[13] == pytest.approx(t2[13], abs=1e-8)

    def test_03_adiabatic(self):
        """Temperature should be equal above and below."""
        xp, yp, zp = 0, 0, 0.001  # meters
        point = grheat.Point(xp, yp, zp, boundary="adiabatic")
        t_pulse = 1
        t = 1.1
        xx, yy = np.meshgrid(np.arange(-0.0005, 0.0005, 0.0001), np.arange(-0.0005, 0.0005, 0.0001))
        t1 = point.pulsed(xx, yy, +0.0001, t, t_pulse)
        t2 = point.pulsed(xx, yy, -0.0001, t, t_pulse)
        assert t1[0, 3] == pytest.approx(t2[0, 3], abs=1e-8)
        assert t1[3, 1] == pytest.approx(t2[3, 1], abs=1e-8)
