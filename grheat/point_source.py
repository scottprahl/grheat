# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=consider-using-f-string
# pylint: disable=no-member
"""
Green's function heat transfer solutions for point source in infinite media.

More documentation at <https://grheat.readthedocs.io>

Typical usage::

    import grheat
    import numpy as np
    import matplotlib.pyplot as plt

    t = np.linspace(0, 500, 100) / 1000   # seconds
    mua = 0.25 * 1000                     # 1/m
    x,y,z = 0,0,0                         # meters
    xp,yp,zp = 0,0,0.001                  # meters
    t_pulse = 0.100                       # seconds

    medium = new grheat.Point()
    T = medium.pulsed(x,y,z,t,xp,yp,zp,t_pulse)

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


class Point:
    """
    Green's function heat transfer solutions for point source in infinite media.

    Typical usage::

        import grheat
        import numpy as np
        import matplotlib.pyplot as plt

        t = np.linspace(0, 500, 100) / 1000   # seconds
        mua = 0.25 * 1000                     # 1/m
        x,y,z = 0,0,0                         # meters
        xp,yp,zp = 0,0,0.001                  # meters
        t_pulse = 0.100                       # seconds

        medium = new grheat.Point()
        T = medium.pulsed(x,y,z,t,xp,yp,zp,t_pulse)

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

    def _instantaneous(self, x, y, z, t, xp, yp, zp, tp):
        """
        Calculate temperature rise due to a 1J instant point source at time t.

        Carslaw and Jaeger page 256, 10.2(2)

        Parameters:
            x, y, z: location for desired temperature [meters]
            t: time of desired temperature [seconds]
            xp, yp, zp: location of point source [meters]
            tp: time of source impulse [seconds]

        Returns:
            normalized temperature
        """
        if t <= tp:
            return 0

        r = np.sqrt((x - xp)**2 + (y - yp)**2 + (z - zp)**2)
        factor = self.capacity * 8 * (np.pi * self.diffusivity * (t - tp))**1.5
        return 1 / factor * np.exp(-r**2 / (4 * self.diffusivity * (t - tp)))

    def instantaneous(self, x, y, z, t, xp, yp, zp, tp):
        """
        Calculate temperature rise due to a 1J instant point source at time(s) t.

        Parameters:
            x, y, z: location for desired temperature [meters]
            t: time(s) of desired temperature [seconds]
            xp, yp, zp: location of point source [meters]
            tp: time of source impulse [seconds]

        Returns:
            Temperature Increase [°C]
        """
        if np.isscalar(t):
            T = self._instantaneous(x, y, z, t, xp, yp, zp, tp)

        else:
            T = np.empty_like(t)
            for i, tt in enumerate(t):
                T[i] = self._instantaneous(x, y, z, tt, xp, yp, zp, tp)
        return T

    def _continuous(self, x, y, z, t, xp, yp, zp):
        """
        Calculate temperature rise of a 1W continuous point source at time t.

        Carslaw and Jaeger page 261, 10.4(2)

        Parameters:
            x, y, z: location for desired temperature [meters]
            t: time of desired temperature [seconds]
            xp, yp, zp: location of point source [meters]

        Returns:
            Temperature Increase [°C]
        """
        if t <= 0:
            return 0

        r = np.sqrt((x - xp)**2 + (y - yp)**2 + (z - zp)**2)
        factor = 1 / self.capacity / (4 * np.pi * self.diffusivity * r)
        T = factor * scipy.special.erfc(r / np.sqrt(4 * self.diffusivity * t))
        return T

    def continuous(self, x, y, z, t, xp, yp, zp):
        """
        Calculate temperature rise of a 1W continuous point source.

        The point source turns on at t=0.

        Parameters:
            x, y, z: location for desired temperature [meters]
            t: time(s) of desired temperature [seconds]
            xp, yp, zp: location of point source [meters]

        Returns:
            Temperature Increase [°C]
        """
        if np.isscalar(t):
            T = self._continuous(x, y, z, t, xp, yp, zp)

        else:
            T = np.empty_like(t)
            for i, tt in enumerate(t):
                T[i] = self._continuous(x, y, z, tt, xp, yp, zp)
        return T

    def _pulsed(self, x, y, z, t, xp, yp, zp, t_pulse):
        """
        Calculate temperature rise due to a 1J pulsed point source at time(s) t.

        1J of heat deposited at (xp, yp, zp) from t=0 to t=t_pulse.

        Parameters:
            x, y, z: location for desired temperature [meters]
            t: time of desired temperature [seconds]
            xp, yp, zp: location of point source [meters]
            t_pulse: duration of pulse [seconds]

        Returns:
            Temperature Increase [°C]
        """
        if t <= 0:
            T = 0
        else:
            T = self._continuous(x, y, z, t, xp, yp, zp)
            if t > t_pulse:
                T -= self._continuous(x, y, z, t - t_pulse, xp, yp, zp)
        return T / t_pulse

    def pulsed(self, x, y, z, t, xp, yp, zp, t_pulse):
        """
        Calculate temperature rise due to a 1J pulsed point source.

        Parameters:
            x, y, z: location for desired temperature [meters]
            t: time(s) of desired temperature [seconds]
            xp, yp, zp: location of point source [meters]
            t_pulse: duration of pulse [seconds]

        Returns
            Temperature Increase [°C]
        """
        if np.isscalar(t):
            T = self._pulsed(x, y, z, t, xp, yp, zp, t_pulse)
        else:
            T = np.empty_like(t)
            for i, tt in enumerate(t):
                T[i] = self._pulsed(x, y, z, tt, xp, yp, zp, t_pulse)
        return T
