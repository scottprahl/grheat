# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=consider-using-f-string
# pylint: disable=no-member
"""
Green's function heat transfer solutions for exponential heating of infinite media.

More documentation at <https://grheat.readthedocs.io>

Typical usage::

    import grheat
    import numpy as np
    import matplotlib.pyplot as plt

    t = np.linspace(0, 500, 100) / 1000   # seconds
    mua = 0.25 * 1000                     # 1/m
    z = 0                                 # meters
    t_pulse = 0.100                       # seconds

    medium = grheat.Absorber(mua)
    T = medium.pulsed(z, t, t_pulse)      # 1 J/m^2
    T *= 1e6                              # 1 J/mm^2

    plt.plot(t * 1000, T, color='blue')
    plt.xlabel("Time (ms)")
    plt.ylabel("Temperature Increase (°C)")
    plt.title("1J/mm² pulse lasting %.0f ms" % t_pulse)
    plt.show()

"""

import scipy.special
import numpy as np

water_heat_capacity = 4.184 * 1e6          # J/degree / m**3
water_thermal_diffusivity = 0.14558 * 1e-6  # m**2/s


class Absorber:
    """
    Green's function heat transfer solutions for exponential heating of infinite media.

    Typical usage::

        import grheat
        import numpy as np
        import matplotlib.pyplot as plt

        t = np.linspace(0, 500, 100) / 1000   # seconds
        mua = 0.25 * 1000                     # 1/m
        z = 0                                 # meters
        t_pulse = 0.100                       # seconds

        medium = new Absorber(mua)
        T = medium.pulsed(z, t, t_pulse)

        plt.plot(t * 1000, T, color='blue')
        plt.xlabel("Time (ms)")
        plt.ylabel("Temperature Increase (°C)")
        plt.title("1J/m² pulse lasting %.0f ms" % t_pulse)
        plt.show()
    """

    def __init__(self,
                 mu,
                 diffusivity=water_thermal_diffusivity,
                 capacity=water_heat_capacity,
                 boundary='infinite'):
        """
        Initialize exponential heating object.

        Parameters:
            mu: exponential attenuation coefficient      [1/meters]
            diffusivity: thermal diffusivity             [m**2/s]
            capacity: volumetric heat capacity           [J/degree/m**3]
            boundary: 'infinite', 'adiabatic', 'zero'
        Returns:
            Planar heat source object
        """
        self.mu = mu                       # 1/meter
        self.diffusivity = diffusivity     # m**2/s
        self.capacity = capacity           # J/degree/kg
        self.boundary = boundary.lower()   # infinite, adiabatic, zero

    def _instantaneous(self, z, t, tp):
        """
        Calculate temperature rise due to 1 J/m² radiant exposure of absorber.

        Volumetric heating decreases as mu*exp(-mu*z) [J/m³].

        Equation follows from integration of instantaneous planar source
        multiplied by exponential over half-space 0<=zp<=Infinity

        Parameters:
            z: depth for desired temperature [meters]
            t: time of desired temperature [seconds]
            tp: time of source impulse [seconds]

        Returns:
            Temperature increase [°C]
        """
        if t <= tp:
            return 0

        tau = self.mu**2 * self.diffusivity * (t - tp)
        zeta = self.mu * z
        factor = self.mu / 2 / self.capacity * np.exp(tau - zeta)
        T = factor * scipy.special.erfc((2 * tau - zeta) / (2 * np.sqrt(tau)))

        if self.boundary != 'infinite':
            factor = self.mu / 2 / self.capacity * np.exp(tau + zeta)
            T1 = factor * scipy.special.erfc((2 * tau + zeta) / (2 * np.sqrt(tau)))

            if self.boundary == 'adiabatic':
                T += T1

            if self.boundary == 'zero':
                T -= T1

        return T

    def instantaneous(self, z, t, tp):
        """
        Calculate temperature rise due to 1 J/m² radiant exposure on absorber.

        Parameters:
            z: depth for desired temperature [meters]
            t: time of desired temperature [seconds]
            tp: time of source impulse [seconds]

        Returns:
            Temperature Increase [°C]
        """
        if np.isscalar(t):
            T = self._instantaneous(z, t, tp)

        else:
            T = np.empty_like(t)
            for i, tt in enumerate(t):
                T[i] = self._instantaneous(z, tt, tp)
        return T

    def _continuous(self, z, t):
        """
        Calculate temperature rise due to 1 W/m² radiant exposure on absorber.

        Volumetric heating decreases as self.mu*exp(-self.mu*z) [W/m³].

        Prahl, "Charts to rapidly estimate temperature following laser irradiation" 1995.

        Parameters:
            z: depth for desired temperature [meters]
            t: time of desired temperature [seconds]
            self.mu: exponential attenuation coefficient [1/meter]

        Returns:
            Temperature increase [°C]
        """
        if t <= 0:
            return 0

        tau = self.mu**2 * self.diffusivity * t
        zeta = self.mu * z
        zz = zeta / np.sqrt(4 * tau)

        T = 2 * np.sqrt(tau / np.pi) * np.exp(-zz**2)
        T += (-1 + zeta) * scipy.special.erfc(-zz)
        T += np.exp(-zz**2) * scipy.special.erfcx(np.sqrt(tau) - zz)

        if self.boundary != 'infinite':
            zeta = -zeta
            zz = zeta / np.sqrt(4 * tau)
            T1 = 2 * np.sqrt(tau / np.pi) * np.exp(-zz**2)
            T1 += (-1 + zeta) * scipy.special.erfc(-zz)
            T1 += np.exp(-zz**2) * scipy.special.erfcx(np.sqrt(tau) - zz)

            if self.boundary == 'adiabatic':
                T += T1

            if self.boundary == 'zero':
                T -= T1

        T /= 2 * self.diffusivity * self.capacity * self.mu
        return T

    def continuous(self, z, t):
        """
        Calculate temperature rise due to 1 J/m² radiant exposure on absorber.

        Volumetric heating decreases as mu*exp(-mu*z) [W/m³].  The
        heating starts at t=0 and continues to t=t.

        Parameters:
            z: depth for desired temperature [meters]
            t: time of desired temperature [seconds]

        Returns:
            Temperature increase [°C]
        """
        if np.isscalar(t):
            T = self._continuous(z, t)

        else:
            T = np.empty_like(t)
            for i, tt in enumerate(t):
                T[i] = self._continuous(z, tt)
        return T

    def _pulsed(self, z, t, t_pulse):
        """
        Calculate temperature rise due to a 1 J/m² pulsed irradiance of absorber.

        Parameters:
            z: depth for desired temperature [meters]
            t: time of desired temperature [seconds]
            t_pulse: duration of pulse [seconds]

        Returns:
            Temperature increase [°C]
        """
        if t <= 0:
            T = 0
        else:
            T = self._continuous(z, t)
            if t > t_pulse:
                T -= self._continuous(z, t - t_pulse)
        return T / t_pulse

    def pulsed(self, z, t, t_pulse):
        """
        Calculate temperature rise due to a 1 J/m² pulsed radiant exposure.

        Pulse lasts from t=0 to t=t_pulse.

        Parameters:
            z: depth for desired temperature [meters]
            t: time(s) of desired temperature [seconds]
            t_pulse: duration of pulse [seconds]

        Returns
            Temperature increase [°C]
        """
        if np.isscalar(t):
            T = self._pulsed(z, t, t_pulse)
        else:
            T = np.empty_like(t)
            for i, tt in enumerate(t):
                T[i] = self._pulsed(z, tt, t_pulse)
        return T
