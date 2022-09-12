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
        z = 0                                 # meters
        zp = 0.001                            # meters
        t_pulse = 0.100                       # seconds

        plane = grheat.Plane(zp)
        T = plane.pulsed(z, t, t_pulse)

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
                 zp,
                 diffusivity=water_thermal_diffusivity,
                 capacity=water_heat_capacity,
                 boundary='infinite'):
        """
        Initialize Planar object.

        Parameters:
            zp: depth of xy-planar source                [meters]
            diffusivity: thermal diffusivity             [m**2/s]
            capacity: heat capacity                      [J/degree/kg]
            boundary: 'infinite', 'adiabatic', 'constant'
        Returns:
            Planar heat source object
        """
        self.zp = zp
        self.diffusivity = diffusivity     # m**2/s
        self.capacity = capacity           # J/degree/kg
        self.boundary = boundary.lower()   # infinite, adiabatic, constant

    def _instantaneous(self, z, t, tp):
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

        r2 = (z - self.zp)**2
        factor = self.capacity * 2 * np.sqrt(np.pi * self.diffusivity * (t - tp))
        T = 1 / factor * np.exp(-r2 / (4 * self.diffusivity * (t - tp)))

        if self.boundary != 'infinite':
            r2 = (z + self.zp)**2
            T1 = 1 / factor * np.exp(-r2 / (4 * self.diffusivity * (t - tp)))

            if self.boundary == 'adiabatic':
                T += T1

            if self.boundary == 'constant':
                T -= T1

        return T

    def instantaneous(self, z, t, tp):
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
            T = self._instantaneous(z, t, tp)

        else:
            T = np.empty_like(t)
            for i, tt in enumerate(t):
                T[i] = self._instantaneous(z, tt, tp)
        return T

    def _continuous(self, z, t):
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

        alpha = np.sqrt((z - self.zp)**2 / (4 * self.diffusivity * t))
        T = np.exp(-alpha**2) / np.sqrt(np.pi) - alpha * scipy.special.erfc(alpha)
        T *= np.sqrt(t / self.diffusivity) / self.capacity

        if self.boundary != 'infinite':
            alpha = np.sqrt((z + self.zp)**2 / (4 * self.diffusivity * t))
            T1 = np.exp(-alpha**2) / np.sqrt(np.pi) - alpha * scipy.special.erfc(alpha)
            T1 *= np.sqrt(t / self.diffusivity) / self.capacity

            if self.boundary == 'adiabatic':
                T += T1

            if self.boundary == 'constant':
                T -= T1

        return T

    def continuous(self, z, t):
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
            T = self._continuous(z, t)

        else:
            T = np.empty_like(t)
            for i, tt in enumerate(t):
                T[i] = self._continuous(z, tt)
        return T

    def _pulsed(self, z, t, t_pulse):
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
            T = self._continuous(z, t)
            if t > t_pulse:
                T -= self._continuous(z, t - t_pulse)
        return T / t_pulse

    def pulsed(self, z, t, t_pulse):
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
            T = self._pulsed(z, t, t_pulse)
        else:
            T = np.empty_like(t)
            for i, tt in enumerate(t):
                T[i] = self._pulsed(z, tt, t_pulse)
        return T
