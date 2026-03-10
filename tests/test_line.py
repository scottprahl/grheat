#! /usr/bin/env python3
# pylint: disable=invalid-name
# pylint: disable=unused-variable
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long

import numpy as np
import pytest

import grheat


class TestInstantaneousLine:
    def test_01_scalar(self):
        """Exercise scalar evaluation for an instantaneous line source."""
        yp, zp = 0, 0.001  # meters
        tp = 0
        line = grheat.Line(yp, zp, tp)
        t = 1
        line.instantaneous(0, 0, t)

    def test_02_time_array(self):
        """Exercise vectorized time evaluation for an instantaneous line source."""
        yp, zp = 0, 0.001  # meters
        tp = 0
        line = grheat.Line(yp, zp, tp)
        t = np.linspace(0, 10)
        line.instantaneous(0, 0, t)

    def test_03_surface(self):
        """Exercise surface-grid evaluation for an instantaneous line source."""
        yp, zp = 0, 0.001  # meters
        tp = 0
        line = grheat.Line(yp, zp, tp)
        t = 1
        yy, zz = np.meshgrid(np.arange(-0.0051, 0.005, 0.001), np.arange(-0.005, 0.005, 0.001))
        line.instantaneous(yy, zz, t)


class TestContinuousLine:
    def test_01_scalar(self):
        """Exercise scalar evaluation for a continuous line source."""
        yp, zp = 0, 0.001  # meters
        line = grheat.Line(yp, zp)
        t = 1
        line.continuous(0, 0, t)

    def test_02_time_array(self):
        """Exercise vectorized time evaluation for a continuous line source."""
        yp, zp = 0, 0.001  # meters
        line = grheat.Line(yp, zp)
        t = np.linspace(0, 10)
        line.continuous(0, 0, t)

    def test_03_surface(self):
        """Exercise surface-grid evaluation for a continuous line source."""
        yp, zp = 0, 0.001  # meters
        line = grheat.Line(yp, zp)
        t = 1
        yy, zz = np.meshgrid(np.arange(-0.0051, 0.005, 0.001), np.arange(-0.005, 0.005, 0.001))
        line.continuous(yy, zz, t)


class TestPulsedLine:
    def test_01_scalar(self):
        """Exercise scalar evaluation for a pulsed line source."""
        yp, zp = 0, 0.001  # meters
        line = grheat.Line(yp, zp)
        t_pulse = 0.5
        t = 1
        line.pulsed(0, 0, t, t_pulse)

    def test_02_time_array(self):
        """Exercise vectorized time evaluation for a pulsed line source."""
        yp, zp = 0, 0.001  # meters
        line = grheat.Line(yp, zp)
        t_pulse = 0.5
        t = np.linspace(0, 10)
        line.pulsed(0, 0, t, t_pulse)

    def test_03_surface(self):
        """Exercise surface-grid evaluation for a pulsed line source."""
        yp, zp = 0, 0.001  # meters
        line = grheat.Line(yp, zp)
        t_pulse = 0.5
        t = 1
        yy, zz = np.meshgrid(np.arange(-0.0051, 0.005, 0.001), np.arange(-0.005, 0.005, 0.001))
        line.pulsed(yy, zz, t, t_pulse)


class TestInstantVsPulsed:
    def test_01_instant(self):
        """Short pulse result should be same as instantaneous source."""
        yp, zp = 0, 0.001  # meters
        tp = 0
        line = grheat.Line(yp, zp, tp)
        t_pulse = 0.00001
        t = 1
        t1 = line.instantaneous(0, 0, t)
        t2 = line.pulsed(0, 0, t, t_pulse)
        assert t1 == pytest.approx(t2, abs=0.001)

    def test_02_instant(self):
        """Short pulse result should be same as instantaneous source."""
        yp, zp = 0, 0.001  # meters
        tp = 0
        line = grheat.Line(yp, zp, tp)
        t_pulse = 0.00001
        t = np.linspace(0, 10)
        t1 = line.instantaneous(0, 0, t)
        t2 = line.pulsed(0, 0, t, t_pulse)
        assert t1[3] == pytest.approx(t2[3], abs=0.001)
        assert t1[13] == pytest.approx(t2[13], abs=0.001)

    def test_03_instant(self):
        """Short pulse result should be same as instantaneous source."""
        yp, zp = 0, 0.001  # meters
        tp = 0
        line = grheat.Line(yp, zp, tp)
        t_pulse = 0.00001
        t = 1
        yy, zz = np.meshgrid(np.arange(-0.0051, 0.005, 0.001), np.arange(-0.005, 0.005, 0.001))

        t1 = line.instantaneous(yy, zz, t)
        t2 = line.pulsed(yy, zz, t, t_pulse)
        assert t1[0, 3] == pytest.approx(t2[0, 3], abs=0.001)
        assert t1[3, 1] == pytest.approx(t2[3, 1], abs=0.001)


class TestConstantBoundary:
    def test_01_zero(self):
        """Surface temperature should be zero."""
        yp, zp = 0.0001, 0.0001  # meters
        line = grheat.Line(yp, zp, boundary="zero")
        t_pulse = 1
        t = 2
        temperature = line.pulsed(0.0001, 0, t, t_pulse)
        assert temperature == 0

    def test_02_zero(self):
        """Surface temperature should be zero at all times."""
        yp, zp = 0.0001, 0.0001  # meters
        line = grheat.Line(yp, zp, boundary="zero")
        t_pulse = 1
        t = np.linspace(0, 10)
        temperature = line.pulsed(0.0001, 0, t, t_pulse)
        assert temperature[3] == 0
        assert temperature[13] == 0


class TestAdiabaticBoundary:
    def test_01_adiabatic(self):
        """Temperature should be equal above and below."""
        yp, zp = 0.0001, 0.0001  # meters
        line = grheat.Line(yp, zp, boundary="adiabatic")
        t_pulse = 1
        t = 2
        t1 = line.pulsed(0, +0.0001, t, t_pulse)
        t2 = line.pulsed(0, -0.0001, t, t_pulse)
        assert t1 == pytest.approx(t2, abs=1e-8)

    def test_02_adiabatic(self):
        """Temperature should be equal above and below at all times."""
        yp, zp = 0.0001, 0.0001  # meters
        line = grheat.Line(yp, zp, boundary="adiabatic")
        t_pulse = 1
        t = np.linspace(0, 2)
        t1 = line.pulsed(0, +0.0001, t, t_pulse)
        t2 = line.pulsed(0, -0.0001, t, t_pulse)
        assert t1[3] == pytest.approx(t2[3], abs=1e-8)
        assert t1[13] == pytest.approx(t2[13], abs=1e-8)
