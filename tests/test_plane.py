#! /usr/bin/env python3
"""Regression tests for plane-source Green's function solutions."""

import numpy as np
import pytest

import grheat


class TestPlaneRepresentation:
    """Check plane-source string representations."""

    def test_str_includes_core_properties(self):
        """Verify ``str(plane)`` includes the key configuration fields."""
        plane = grheat.Plane(zp=0.003, tp=0.25, boundary="adiabatic")

        description = str(plane)

        assert "Plane Properties:" in description
        assert "zp: 0.003 meters" in description
        assert "tp: 0.25 seconds" in description
        assert "boundary: adiabatic" in description


class TestInstantaneousPlane:
    """Check instantaneous plane-source solutions."""

    def test_01_scalar(self):
        """Exercise scalar evaluation for an instantaneous planar source."""
        zp = 0.001  # meters
        tp = 0
        t = 1
        plane = grheat.Plane(zp, tp)
        plane.instantaneous(0, t)

    def test_02_time_array(self):
        """Exercise vectorized time evaluation for an instantaneous planar source."""
        zp = 0.001  # meters
        tp = 0
        t = np.linspace(0, 10)
        plane = grheat.Plane(zp, tp)
        plane.instantaneous(0, t)


class TestContinuousPlane:
    """Check continuous plane-source solutions."""

    def test_01_scalar(self):
        """Exercise scalar evaluation for a continuous planar source."""
        zp = 0.001  # meters
        t = 1
        plane = grheat.Plane(zp)
        plane.continuous(0, t)

    def test_02_time_array(self):
        """Exercise vectorized time evaluation for a continuous planar source."""
        zp = 0.001  # meters
        t = np.linspace(0, 10)
        plane = grheat.Plane(zp)
        plane.continuous(0, t)


class TestPulsedPlane:
    """Check pulsed plane-source solutions."""

    def test_01_scalar(self):
        """Exercise scalar evaluation for a pulsed planar source."""
        zp = 0.001  # meters
        t_pulse = 0.5
        t = 1
        plane = grheat.Plane(zp)
        plane.pulsed(0, t, t_pulse)

    def test_02_time_array(self):
        """Exercise vectorized time evaluation for a pulsed planar source."""
        zp = 0.001  # meters
        t_pulse = 0.5
        t = np.linspace(0, 10)
        plane = grheat.Plane(zp)
        plane.pulsed(0, t, t_pulse)


class TestInstantVsPulsed:
    """Compare short pulses against instantaneous plane-source results."""

    def test_01_instant(self):
        """Short pulse result should be same as instantaneous source."""
        zp = 0.001  # meters
        tp = 0
        t_pulse = 0.00001
        t = 1
        radiant_exposure = 1e6  # 1 J/mm² in J/m²
        plane = grheat.Plane(zp, tp)
        t1 = radiant_exposure * plane.instantaneous(0, t)
        t2 = radiant_exposure * plane.pulsed(0, t, t_pulse)
        assert t1 == pytest.approx(t2, abs=0.001)

    def test_02_instant(self):
        """Short pulse result should be same as instantaneous source."""
        zp = 0.001  # meters
        t_pulse = 0.00001
        tp = 0
        t = np.linspace(0, 10)
        radiant_exposure = 1e6  # 1 J/mm² in J/m²
        plane = grheat.Plane(zp, tp)
        t1 = radiant_exposure * plane.instantaneous(0, t)
        t2 = radiant_exposure * plane.pulsed(0, t, t_pulse)
        assert t1[3] == pytest.approx(t2[3], abs=0.001)
        assert t1[13] == pytest.approx(t2[13], abs=0.001)


class TestConstantBoundary:
    """Check zero-temperature boundary behavior for plane sources."""

    def test_01_zero(self):
        """Surface temperature should be zero."""
        zp = 0.001  # meters
        plane = grheat.Plane(zp, boundary="zero")
        t_pulse = 1
        t = 2
        temperature = plane.pulsed(0, t, t_pulse)
        assert temperature == 0

    def test_02_zero(self):
        """Surface temperature should be zero at all times."""
        zp = 0.001  # meters
        plane = grheat.Plane(zp, boundary="zero")
        t_pulse = 1
        t = np.linspace(0, 10)
        temperature = plane.pulsed(0, t, t_pulse)
        assert temperature[3] == 0
        assert temperature[13] == 0


class TestAdiabaticBoundary:
    """Check adiabatic boundary symmetry for plane sources."""

    def test_01_adiabatic(self):
        """Temperature should be equal above and below."""
        zp = 0.001  # meters
        plane = grheat.Plane(zp, boundary="adiabatic")
        t_pulse = 1
        t = 2
        t1 = plane.pulsed(+0.0001, t, t_pulse)
        t2 = plane.pulsed(-0.0001, t, t_pulse)
        assert t1 == pytest.approx(t2, abs=1e-8)

    def test_02_adiabatic(self):
        """Temperature should be equal above and below at all times."""
        zp = 0.001  # meters
        plane = grheat.Plane(zp, boundary="adiabatic")
        t_pulse = 1
        t = np.linspace(0, 2)
        t1 = plane.pulsed(+0.0001, t, t_pulse)
        t2 = plane.pulsed(-0.0001, t, t_pulse)
        assert t1[3] == pytest.approx(t2[3], abs=1e-8)
        assert t1[13] == pytest.approx(t2[13], abs=1e-8)
