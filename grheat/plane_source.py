# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=consider-using-f-string
# pylint: disable=no-member
"""
Green's function heat transfer solutions for xy-planar source in infinite media.

More documentation at <https://grheat.readthedocs.io>

Typical usage::

    Typical usage::

        import grheat
        import numpy as np
        import matplotlib.pyplot as plt

        t = np.linspace(0, 500, 100) / 1000   # seconds
        mua = 0.25 * 1000                     # 1/m
        z = 0                                 # meters
        zp = 0.001                            # meters
        t_pulse = 0.100                       # seconds

        plane = grheat.Plane()
        T = plane.pulsed(x,y,z,t,xp,yp,zp,t_pulse)

        plt.plot(t * 1000, T, color='blue')
        plt.xlabel("Time (ms)")
        plt.ylabel("Temperature Increase (°C)")
        plt.title("1J pulse lasting %.0f ms" % t_pulse)
        plt.show()
"""

import scipy.special
import numpy as np

water_heat_capacity = 4.184 * 1e6           # J/degree / m**3
water_thermal_diffusivity = 0.14558 * 1e-6  # m**2/s


class Plane:
    """
    Green's function heat transfer solutions for point source in infinite media.

    Typical usage::

        import grheat
        import numpy as np
        import matplotlib.pyplot as plt

        t = np.linspace(0, 500, 100) / 1000   # seconds
        mua = 0.25 * 1000                     # 1/m
        z = 0                                 # meters
        zp = 0.001                            # meters
        t_pulse = 0.100                       # seconds

        plane = grheat.Plane()
        T = plane.pulsed(x,y,z,t,xp,yp,zp,t_pulse)

        plt.plot(t * 1000, T, color='blue')
        plt.xlabel("Time (ms)")
        plt.ylabel("Temperature Increase (°C)")
        plt.title("1J pulse lasting %.0f ms" % t_pulse)
        plt.show()
    """

    def __init__(self,
                 diffusivity=water_thermal_diffusivity,
                 capacity=water_heat_capacity,
                 boundary='infinite'):
        self.diffusivity = diffusivity     # m**2/s
        self.capacity = capacity           # J/degree/kg
        self.boundary = boundary.lower()   # infinite, adiabatic, constant

    def _instantaneous(self, z, t, zp, tp):
        """
        Calculate temperature rise due to a 1 J/m² instantaneous xy-planar source at time t.

        The plane is parallel to the surface and passes through zp.

        Carslaw and Jaeger page 259, 10.3(4)

        Parameters:
            z: depth of desired temperature [meters]
            t: time of desired temperature [seconds]
            zp: depth of xy-planar source [meters]
            tp: time of source impulse [seconds]

        Returns:
            Temperature increase [°C]
        """
        if t <= tp:
            return 0

        r = np.abs(z - zp)
        factor = self.capacity * 2 * np.sqrt(np.pi * self.diffusivity * (t - tp))
        return 1 / factor * np.exp(-r**2 / (4 * self.diffusivity * (t - tp)))

    def instantaneous(self, z, t, zp, tp):
        """
        Calculate temperature rise due to a 1 J/m² instant xy-planar source.

        The plane is parallel to the surface and passes through zp.

        Parameters:
            z: depth for desired temperature [meters]
            t: time of desired temperature [seconds]
            zp: depth of xy-planar source [meters]
            tp: time of source impulse [seconds]

        Returns:
            Temperature increase [°C]
        """
        if np.isscalar(t):
            T = self._instantaneous(z, t, zp, tp)

        else:
            T = np.empty_like(t)
            for i, tt in enumerate(t):
                T[i] = self._instantaneous(z, tt, zp, tp)
        return T

    def _continuous(self, z, t, zp):
        """
        Calculate temperature rise due to a 1 W/m² xy-planar source.

        The xy-planar source is located at a depth zp and is turned on at t=0.
        It remains on until t=t.

        Equation obtained by integrating planar Green's function from 0 to t in
        Mathematica.

        Parameters:
            z: depth for desired temperature [meters]
            t: time of desired temperature [seconds]
            zp: depth of xy-planar source [meters]

        Returns:
            Temperature increase [°C]
        """
        if t <= 0:
            return 0

        alpha = np.sqrt((z - zp)**2 / (4 * self.diffusivity * t))
        T = np.exp(-alpha**2) / np.sqrt(np.pi) - alpha * scipy.special.erfc(alpha)
        return np.sqrt(t / self.diffusivity) / self.capacity * T

    def continuous(self, z, t, zp):
        """
        Calculate temperature rise due to a 1W/m² continuous xy-planar source.

        The xy-planar source turns on at t=0 and passes through zp.

        Parameters:
            z: depth for desired temperature [meters]
            t: time(s) of desired temperature [seconds]
            zp: depth of xy-planar source [meters]

        Returns:
            Temperature increase [°C]
        """
        if np.isscalar(t):
            T = self._continuous(z, t, zp)

        else:
            T = np.empty_like(t)
            for i, tt in enumerate(t):
                T[i] = self._continuous(z, tt, zp)
        return T

    def _pulsed(self, z, t, zp, t_pulse):
        """
        Calculate temperature rise due to a 1 J/m² pulsed xy-planar source at time t.

        1 J/m² of heat deposited in an xy-plane passing through (zp) from t=0 to t=t_pulse.

        Parameters:
            z: depth for desired temperature [meters]
            t: time of desired temperature [seconds]
            zp: depth of xy-planar source [meters]
            t_pulse: duration of pulse [seconds]

        Returns:
            Temperature increase [°C]
        """
        if t <= 0:
            T = 0
        else:
            T = self._continuous(z, t, zp)
            if t > t_pulse:
                T -= self._continuous(z, t - t_pulse, zp)
        return T / t_pulse

    def pulsed(self, z, t, zp, t_pulse):
        """
        Calculate temperature rise due to a 1 J/m² pulsed xy-planar source.

        Parameters:
            z: depth for desired temperature [meters]
            t: time(s) of desired temperature [seconds]
            zp: depth of xy-planar source [meters]
            t_pulse: duration of pulse [seconds]

        Returns
            Temperature increase [°C]
        """
        if np.isscalar(t):
            T = self._pulsed(z, t, zp, t_pulse)
        else:
            T = np.empty_like(t)
            for i, tt in enumerate(t):
                T[i] = self._pulsed(z, tt, zp, t_pulse)
        return T
