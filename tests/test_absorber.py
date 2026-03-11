"""Tests for the Absorber class."""

import numpy as np
import pytest

import grheat

joules_per_calorie = 4.184


class TestAbsorberRepresentation:
    """Check absorber string representations."""

    def test_str_includes_core_properties(self):
        """Verify ``str(absorber)`` includes the key configuration fields."""
        absorber = grheat.Absorber(mu_a=250, tp=0.25, boundary="adiabatic")

        description = str(absorber)

        assert "Absorber Properties:" in description
        assert "mu_a: 250 1/meters" in description
        assert "tp: 0.25 seconds" in description
        assert "boundary: adiabatic" in description


class TestInstantaneousAbsorber:
    """Instant absorber tests."""

    def test_01_scalar(self):
        """Does it work for scalar values."""
        mua = 1000  # 1/meter
        tp = 0
        t = 1
        medium = grheat.Absorber(mua, tp)
        medium.instantaneous(0, t)

    def test_02_time_array(self):
        """Does it work for a time array."""
        mua = 1000  # 1/meter
        tp = 0
        t = np.linspace(0, 10)
        medium = grheat.Absorber(mua, tp)
        medium.instantaneous(0, t)

    def test_03_tp_array(self):
        """Does it work for a pulse time array."""
        mua = 1000  # 1/meter
        t = 1
        tp = np.linspace(0, 10, 11)
        medium = grheat.Absorber(mua, tp)
        medium.instantaneous(0, t)

    def test_04_z_array(self):
        """Does it work for a z array."""
        mua = 1000  # 1/meter
        t = 1
        tp = 0
        z = np.linspace(0, 10, 11)
        medium = grheat.Absorber(mua, tp)
        medium.instantaneous(z, t)

    def test_05_total_energy(self):
        """Test for total energy."""
        mu_a = 1 * 1000  # 1 / m
        medium = grheat.Absorber(mu_a)
        z = np.linspace(-0.0015, 0.006, 4001)
        dz = z[1] - z[0]

        for t in [0, 0.01, 0.1, 1]:
            temperature = medium.instantaneous(z, t) * 1e6
            total = temperature.sum() * dz * joules_per_calorie
            assert total == pytest.approx(1, abs=0.01)


class TestContinuousAbsorber:
    """Continuous absorber tests."""

    def test_01_scalar(self):
        """Does it work for scalar values."""
        mua = 1000  # 1/meter
        t = 1
        medium = grheat.Absorber(mua)
        medium.continuous(0, t)

    def test_02_time_array(self):
        """Ensure works with arrays of time."""
        mua = 1000  # 1/meter
        t = np.linspace(0, 10)
        medium = grheat.Absorber(mua)
        medium.continuous(0, t)

    def test_03_tp_array(self):
        """Ensure works with arrays of pulses."""
        mua = 1000  # 1/meter
        t = 1
        tp = np.linspace(0, 10, 11)
        medium = grheat.Absorber(mua, tp)
        medium.continuous(0, t)

    def test_04_z_array(self):
        """Ensure works with arrays of depths."""
        mua = 1000  # 1/meter
        t = 1
        z = np.linspace(0, 10, 11)
        medium = grheat.Absorber(mua)
        medium.continuous(z, t)

    def test_05_energy(self):
        """Ensure total energy delivered is correct."""
        mu_a = 1 * 1000  # 1 / m
        medium = grheat.Absorber(mu_a)
        z = np.linspace(-0.0015, 0.006, 401)
        dz = z[1] - z[0]

        for t in [0, 0.01, 0.1, 1]:
            temperature = medium.continuous(z, t) * 1e6
            total = temperature.sum() * dz * joules_per_calorie
            assert total == pytest.approx(t, abs=0.01)


class TestPulsedAbsorber:
    """Pulsed absorber tests."""

    def test_01_scalar(self):
        """Test with all scalar inputs."""
        mua = 1000  # 1/meter
        t_pulse = 0.5
        t = 1
        medium = grheat.Absorber(mua)
        medium.pulsed(0, t, t_pulse)

    def test_02_time_array(self):
        """Test with an array of time."""
        mua = 1000  # 1/meter
        t_pulse = 0.5
        t = np.linspace(0, 10)
        medium = grheat.Absorber(mua)
        medium.pulsed(0, t, t_pulse)

    def test_03_total_energy(self):
        """Total energy at end of pulse."""
        mu_a = 1 * 1000  # 1 / m
        medium = grheat.Absorber(mu_a)
        z = np.linspace(-0.0015, 0.006, 401)
        dz = z[1] - z[0]

        for t in [0.001, 0.01, 0.1, 1]:
            temperature = medium.pulsed(z, t, t) * 1e6
            total = temperature.sum() * dz * joules_per_calorie
            assert total == pytest.approx(1, abs=0.01)

    def test_04_total_energy(self):
        """Total energy during and after pulse."""
        mu_a = 1 * 1000  # 1 / m
        medium = grheat.Absorber(mu_a)
        z = np.linspace(-0.003, 0.008, 4001)
        dz = z[1] - z[0]
        t_pulse = 0.1

        for t, expected in [(0.5 * t_pulse, 0.5), (2 * t_pulse, 1), (10 * t_pulse, 1)]:
            temperature = medium.pulsed(z, t, t_pulse) * 1e6
            total = temperature.sum() * dz * joules_per_calorie
            assert total == pytest.approx(expected, abs=0.01)


class TestInstantVsPulsed:
    """Instant and Pulsed absorber tests."""

    def test_01_instant(self):
        """Short pulse result should be same as instantaneous source."""
        mua = 200  # 1/meter
        t_pulse = 0.00001
        tp = 0
        t = 0.5
        z = 0.001
        radiant_exposure = 1e6  # 1 J/mm² in J/m²
        medium = grheat.Absorber(mua, tp)
        t1 = radiant_exposure * medium.instantaneous(z, t)
        t2 = radiant_exposure * medium.pulsed(z, t, t_pulse)
        assert t1 == pytest.approx(t2, abs=0.001)

    def test_02_instant(self):
        """Short pulse result should be same as instantaneous source."""
        mua = 200  # 1/meter
        t_pulse = 0.00001
        tp = 0
        t = 2
        z = 0.001
        radiant_exposure = 1e6  # 1 J/mm² in J/m²
        medium = grheat.Absorber(mua, tp)
        t1 = radiant_exposure * medium.instantaneous(z, t)
        t2 = radiant_exposure * medium.pulsed(z, t, t_pulse)
        assert t1 == pytest.approx(t2, abs=0.001)

    def test_03_instant(self):
        """Short pulse result should be same as instantaneous source."""
        mua = 100  # 1/meter
        t_pulse = 0.0001
        tp = 0
        t = np.linspace(0, 10)
        radiant_exposure = 1e6  # 1 J/mm² in J/m²
        medium = grheat.Absorber(mua, tp)
        t1 = radiant_exposure * medium.instantaneous(0, t)
        t2 = radiant_exposure * medium.pulsed(0, t, t_pulse)
        assert t1[3] == pytest.approx(t2[3], abs=0.001)
        assert t1[13] == pytest.approx(t2[13], abs=0.001)


class TestIntegratedPlane:
    """Integrated plane absorber tests."""

    def test_01_instant(self):
        """Matches numerical integration of plane sources."""
        mua = 100  # 1/meter
        tp = 0
        t = 0.5
        z = 0.001
        radiant_exposure = 1e6  # 1 J/mm² in J/m²
        medium = grheat.Absorber(mua, tp)
        t1 = radiant_exposure * medium.instantaneous(z, t)
        zp_array = np.linspace(0, 20 * z, 1000)
        total = 0
        plane = grheat.Plane(0)
        for zp in zp_array:
            plane.zp = zp
            total += np.exp(-mua * zp) * plane.instantaneous(z, t)
        t2 = radiant_exposure * total * mua * (zp_array[1] - zp_array[0])
        assert t1 == pytest.approx(t2, abs=0.02)

    def test_02_pulsed(self):
        """Matches numerical integration of plane sources."""
        mua = 100  # 1/meter
        tpulse = 0.1
        t = 0.2
        z = 0.001
        radiant_exposure = 1e6  # 1 J/mm² in J/m²
        medium = grheat.Absorber(mua)
        t1 = radiant_exposure * medium.pulsed(z, t, tpulse)
        zp_array = np.linspace(0, 20 * z, 1000)
        total = 0
        plane = grheat.Plane(0)
        for zp in zp_array:
            plane.zp = zp
            total += np.exp(-mua * zp) * plane.pulsed(z, t, tpulse)
        t2 = radiant_exposure * total * mua * (zp_array[1] - zp_array[0])
        assert t1 == pytest.approx(t2, abs=0.02)


class TestConstantBoundary:
    """Constant Boundary tests."""

    def test_01_zero(self):
        """Surface temperature should be zero."""
        mua = 1000  # 1/meter
        medium = grheat.Absorber(mua, boundary="zero")
        t_pulse = 1
        t = 2
        temperature = medium.pulsed(0, t, t_pulse)
        assert temperature == 0

    def test_02_zero(self):
        """Surface temperature should be zero at all times."""
        mua = 1000  # 1/meter
        medium = grheat.Absorber(mua, boundary="zero")
        t_pulse = 1
        t = np.linspace(0, 10)
        temperature = medium.pulsed(0, t, t_pulse)
        assert temperature[3] == 0
        assert temperature[13] == 0


class TestAdiabaticBoundary:
    """Adiabatic Boundary tests."""

    def test_01_adiabatic(self):
        """Temperature should be equal above and below."""
        mua = 1000  # 1/meter
        medium = grheat.Absorber(mua, boundary="adiabatic")
        t_pulse = 1
        t = 2
        t1 = medium.pulsed(+0.0001, t, t_pulse)
        t2 = medium.pulsed(-0.0001, t, t_pulse)
        assert t1 == pytest.approx(t2, abs=1e-3)

    def test_02_adiabatic(self):
        """Temperature should be equal above and below at all times."""
        mua = 1000  # 1/meter
        medium = grheat.Absorber(mua, boundary="adiabatic")
        t_pulse = 1
        t = np.linspace(0, 2)
        t1 = medium.pulsed(+0.0001, t, t_pulse)
        t2 = medium.pulsed(-0.0001, t, t_pulse)
        assert t1[3] == pytest.approx(t2[3], abs=1e-3)
        assert t1[13] == pytest.approx(t2[13], abs=1e-3)

    def test_03_adiabatic(self):
        """Total Temperature should be equal to 1."""
        mu_a = 1 * 1000  # 1 / m
        medium = grheat.Absorber(mu_a, boundary="adiabatic")
        z = np.linspace(0, 0.010, 5001)
        dz = z[1] - z[0]

        for t in [0, 0.01, 0.1, 1]:
            temperature = medium.instantaneous(z, t) * 1e6
            total = temperature.sum() * dz * joules_per_calorie
            assert total == pytest.approx(1, abs=0.01)
