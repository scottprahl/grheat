"""Tests for the Absorber class."""

import unittest
import numpy as np
import grheat
joules_per_calorie = 4.184


class InstantaneousAbsorber(unittest.TestCase):
    """Instant absorber tests."""

    def test_01_scalar(self):
        """Does it work for scalar values."""
        mua = 1000  # 1/meter
        tp = 0
        t = 1
        medium = grheat.Absorber(mua, tp)
        _T = medium.instantaneous(0, t)

    def test_02_time_array(self):
        """Does it work for a time array."""
        mua = 1000  # 1/meter
        tp = 0
        t = np.linspace(0, 10)
        medium = grheat.Absorber(mua, tp)
        _T = medium.instantaneous(0, t)

    def test_03_tp_array(self):
        """Does it work for a pulse time array."""
        mua = 1000  # 1/meter
        t = 1
        tp = np.linspace(0, 10, 11)
        medium = grheat.Absorber(mua, tp)
        _T = medium.instantaneous(0, t)

    def test_04_z_array(self):
        """Does it work for a z array."""
        mua = 1000  # 1/meter
        t = 1
        tp = 0
        z = np.linspace(0, 10, 11)
        medium = grheat.Absorber(mua, tp)
        _T = medium.instantaneous(z, t)

    def test_05_total_energy(self):
        """Test for total energy."""
        mu_a = 1 * 1000  # 1 / m
        medium = grheat.Absorber(mu_a)
        z = np.linspace(-0.0015, 0.006, 4001)
        dz = z[1] - z[0]

        t = 0
        T = medium.instantaneous(z, t) * 1e6
        total = T.sum() * dz * joules_per_calorie
        self.assertAlmostEqual(1, total, delta=0.01)

        t = 0.01
        T = medium.instantaneous(z, t) * 1e6
        total = T.sum() * dz * joules_per_calorie
        self.assertAlmostEqual(1, total, delta=0.01)

        t = 0.1
        T = medium.instantaneous(z, t) * 1e6
        total = T.sum() * dz * joules_per_calorie
        self.assertAlmostEqual(1, total, delta=0.01)

        t = 1
        T = medium.instantaneous(z, t) * 1e6
        total = T.sum() * dz * joules_per_calorie
        self.assertAlmostEqual(1, total, delta=0.01)


class ContinuousAbsorber(unittest.TestCase):
    """Continuous absorber tests."""

    def test_01_scalar(self):
        """Does it work for scalar values."""
        mua = 1000  # 1/meter
        t = 1
        medium = grheat.Absorber(mua)
        _T = medium.continuous(0, t)

    def test_02_time_array(self):
        """Ensure works with arrays of time."""
        mua = 1000  # 1/meter
        t = np.linspace(0, 10)
        medium = grheat.Absorber(mua)
        _T = medium.continuous(0, t)

    def test_03_tp_array(self):
        """Ensure works with arrays of pulses."""
        mua = 1000  # 1/meter
        t = 1
        tp = np.linspace(0, 10, 11)
        medium = grheat.Absorber(mua, tp)
        _T = medium.continuous(0, t)

    def test_04_z_array(self):
        """Ensure works with arrays of depths."""
        mua = 1000  # 1/meter
        t = 1
        z = np.linspace(0, 10, 11)
        medium = grheat.Absorber(mua)
        _T = medium.continuous(z, t)

    def test_05_energy(self):
        """Ensure total energy delivered is correct."""
        mu_a = 1 * 1000  # 1 / m
        medium = grheat.Absorber(mu_a)
        z = np.linspace(-0.0015, 0.006, 401)
        dz = z[1] - z[0]

        print('No surface boundary condition, instantaneous pulse')
        print('time   total')
        t = 0
        T = medium.continuous(z, t) * 1e6
        total = T.sum() * dz * joules_per_calorie
        self.assertAlmostEqual(t, total, delta=0.01)

        t = 0.01
        T = medium.continuous(z, t) * 1e6
        total = T.sum() * dz * joules_per_calorie
        self.assertAlmostEqual(t, total, delta=0.01)

        t = 0.1
        T = medium.continuous(z, t) * 1e6
        total = T.sum() * dz * joules_per_calorie
        self.assertAlmostEqual(t, total, delta=0.01)

        t = 1
        T = medium.continuous(z, t) * 1e6
        total = T.sum() * dz * joules_per_calorie
        self.assertAlmostEqual(t, total, delta=0.01)


class PulsedAbsorber(unittest.TestCase):
    """Pulsed absorber tests."""

    def test_01_scalar(self):
        """Test with all scalar inputs."""
        mua = 1000  # 1/meter
        t_pulse = 0.5
        t = 1
        medium = grheat.Absorber(mua)
        _T = medium.pulsed(0, t, t_pulse)

    def test_02_time_array(self):
        """Test with an array of time."""
        mua = 1000  # 1/meter
        t_pulse = 0.5
        t = np.linspace(0, 10)
        medium = grheat.Absorber(mua)
        _T = medium.pulsed(0, t, t_pulse)

    def test_03_total_energy(self):
        """Total energy at end of pulse."""
        mu_a = 1 * 1000  # 1 / m
        medium = grheat.Absorber(mu_a)
        z = np.linspace(-0.0015, 0.006, 401)
        dz = z[1] - z[0]

        t = 0.001
        t_pulse = t
        T = medium.pulsed(z, t, t_pulse) * 1e6
        total = T.sum() * dz * joules_per_calorie
        self.assertAlmostEqual(1, total, delta=0.01)

        t = 0.01
        t_pulse = t
        T = medium.pulsed(z, t, t_pulse) * 1e6
        total = T.sum() * dz * joules_per_calorie
        self.assertAlmostEqual(1, total, delta=0.01)

        t = 0.1
        t_pulse = t
        T = medium.pulsed(z, t, t_pulse) * 1e6
        total = T.sum() * dz * joules_per_calorie
        self.assertAlmostEqual(1, total, delta=0.01)

        t = 1
        t_pulse = t
        T = medium.pulsed(z, t, t_pulse) * 1e6
        total = T.sum() * dz * joules_per_calorie
        self.assertAlmostEqual(1, total, delta=0.01)

    def test_04_total_energy(self):
        """Total energy during and after pulse."""
        mu_a = 1 * 1000  # 1 / m
        medium = grheat.Absorber(mu_a)
        z = np.linspace(-0.003, 0.008, 4001)
        dz = z[1] - z[0]
        t_pulse = 0.1

        t = 0.5 * t_pulse
        T = medium.pulsed(z, t, t_pulse) * 1e6
        total = T.sum() * dz * joules_per_calorie
        self.assertAlmostEqual(0.5, total, delta=0.01)

        t = 2 * t_pulse
        T = medium.pulsed(z, t, t_pulse) * 1e6
        total = T.sum() * dz * joules_per_calorie
        self.assertAlmostEqual(1, total, delta=0.01)

        t = 10 * t_pulse
        T = medium.pulsed(z, t, t_pulse) * 1e6
        total = T.sum() * dz * joules_per_calorie
        self.assertAlmostEqual(1, total, delta=0.01)


class InstantVsPulsed(unittest.TestCase):
    """Instant and Pulsed absorber tests."""

    def test_01_instant(self):
        """Short pulse result should be same as instantaneous source."""
        mua = 200  # 1/meter
        t_pulse = 0.00001
        tp = 0
        t = 0.5
        z = 0.001
        radiant_exposure = 1e6    # 1 J/mm² in J/m²
        medium = grheat.Absorber(mua, tp)
        T1 = radiant_exposure * medium.instantaneous(z, t)
        T2 = radiant_exposure * medium.pulsed(z, t, t_pulse)
        self.assertAlmostEqual(T1, T2, delta=0.001)

    def test_02_instant(self):
        """Short pulse result should be same as instantaneous source."""
        mua = 200  # 1/meter
        t_pulse = 0.00001
        tp = 0
        t = 2
        z = 0.001
        radiant_exposure = 1e6    # 1 J/mm² in J/m²
        medium = grheat.Absorber(mua, tp)
        T1 = radiant_exposure * medium.instantaneous(z, t)
        T2 = radiant_exposure * medium.pulsed(z, t, t_pulse)
        self.assertAlmostEqual(T1, T2, delta=0.001)

    def test_03_instant(self):
        """Short pulse result should be same as instantaneous source."""
        mua = 100  # 1/meter
        t_pulse = 0.0001
        tp = 0
        t = np.linspace(0, 10)
        radiant_exposure = 1e6    # 1 J/mm² in J/m²
        medium = grheat.Absorber(mua, tp)
        T1 = radiant_exposure * medium.instantaneous(0, t)
        T2 = radiant_exposure * medium.pulsed(0, t, t_pulse)
        self.assertAlmostEqual(T1[3], T2[3], delta=0.001)
        self.assertAlmostEqual(T1[13], T2[13], delta=0.001)


class IntegratedPlane(unittest.TestCase):
    """Integrated plane absorber tests."""

    def test_01_instant(self):
        """Matches numerical integration of plane sources."""
        mua = 100  # 1/meter
        tp = 0
        t = 0.5
        z = 0.001
        radiant_exposure = 1e6    # 1 J/mm² in J/m²
        medium = grheat.Absorber(mua, tp)
        T1 = radiant_exposure * medium.instantaneous(z, t)
        zp_array = np.linspace(0, 20 * z, 1000)
        total = 0
        plane = grheat.Plane(0)
        for i, zp in enumerate(zp_array):
            plane.zp = zp
            total += np.exp(-mua * zp) * plane.instantaneous(z, t)
        T2 = radiant_exposure * total * mua * (zp_array[1] - zp_array[0])
#        print(T1 / T2)
#        print(T2 / T1)
        self.assertAlmostEqual(T1, T2, delta=0.02)

    def test_02_pulsed(self):
        """Matches numerical integration of plane sources."""
        mua = 100  # 1/meter
        tpulse = 0.1
        t = 0.2
        z = 0.001
        radiant_exposure = 1e6    # 1 J/mm² in J/m²
        medium = grheat.Absorber(mua)
        T1 = radiant_exposure * medium.pulsed(z, t, tpulse)
        zp_array = np.linspace(0, 20 * z, 1000)
        total = 0
        plane = grheat.Plane(0)
        for i, zp in enumerate(zp_array):
            plane.zp = zp
            total += np.exp(-mua * zp) * plane.pulsed(z, t, tpulse)
        T2 = radiant_exposure * total * mua * (zp_array[1] - zp_array[0])
#        print(T1 / T2)
#        print(T2 / T1)
        self.assertAlmostEqual(T1, T2, delta=0.02)


class ConstantBoundary(unittest.TestCase):
    """Constant Boundary tests."""

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
        t = np.linspace(0, 10)
        T = medium.pulsed(0, t, t_pulse)
        self.assertEqual(T[3], 0)
        self.assertEqual(T[13], 0)


class AdiabaticBoundary(unittest.TestCase):
    """Adiabatic Boundary tests."""

    def test_01_adiabatic(self):
        """Temperature should be equal above and below."""
        mua = 1000  # 1/meter
        medium = grheat.Absorber(mua, boundary='adiabatic')
        t_pulse = 1
        t = 2
        T1 = medium.pulsed(+0.0001, t, t_pulse)
        T2 = medium.pulsed(-0.0001, t, t_pulse)
        self.assertAlmostEqual(T1, T2, delta=1e-3)

    def test_02_adiabatic(self):
        """Temperature should be equal above and below at all times."""
        mua = 1000  # 1/meter
        medium = grheat.Absorber(mua, boundary='adiabatic')
        t_pulse = 1
        t = np.linspace(0, 2)
        T1 = medium.pulsed(+0.0001, t, t_pulse)
        T2 = medium.pulsed(-0.0001, t, t_pulse)
        self.assertAlmostEqual(T1[3], T2[3], delta=1e-3)
        self.assertAlmostEqual(T1[13], T2[13], delta=1e-3)

    def test_03_adiabatic(self):
        """Total Temperature should be equal to 1."""
        mu_a = 1 * 1000  # 1 / m
        medium = grheat.Absorber(mu_a, boundary='adiabatic')
        z = np.linspace(0, 0.010, 5001)
        dz = z[1] - z[0]

        t = 0
        T = medium.instantaneous(z, t) * 1e6
        total = T.sum() * dz * joules_per_calorie
        self.assertAlmostEqual(1, total, delta=0.01)

        t = 0.01
        T = medium.instantaneous(z, t) * 1e6
        total = T.sum() * dz * joules_per_calorie
        self.assertAlmostEqual(1, total, delta=0.01)

        t = 0.1
        T = medium.instantaneous(z, t) * 1e6
        total = T.sum() * dz * joules_per_calorie
        self.assertAlmostEqual(1, total, delta=0.01)

        t = 1
        T = medium.instantaneous(z, t) * 1e6
        total = T.sum() * dz * joules_per_calorie
        self.assertAlmostEqual(1, total, delta=0.01)


if __name__ == '__main__':
    unittest.main()
