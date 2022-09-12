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
    x,y,z = 0,0,0                         # meters
    xp,yp,zp = 0,0,0.001                  # meters
    t_pulse = 0.100                       # seconds

    medium = new grheat.Point(xp, yp, zp)
    T = medium.pulsed(x, y, z, t, t_pulse)

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

        medium = new grheat.Point(xp, yp, zp)
        T = medium.pulsed(x,y,z,t,t_pulse)

        plt.plot(t * 1000, T, color='blue')
        plt.xlabel("Time (ms)")
        plt.ylabel("Temperature Increase (°C)")
        plt.title("1J pulse lasting %.0f ms" % t_pulse)
        plt.show()
    """

    def __init__(self,
                 xp, yp, zp,
                 diffusivity=water_thermal_diffusivity,
                 capacity=water_heat_capacity,
                 boundary='infinite'):
        """
        Initialize Point object.

        Parameters:
            xp: x location of source                     [meters]
            yp: y location of source                     [meters]
            zp: z location of source                     [meters]
            diffusivity: thermal diffusivity             [m**2/s]
            capacity: heat capacity                      [J/degree/kg]
            boundary: 'infinite', 'adiabatic', 'constant'
        Returns:
            Planar heat source object
        """
        self.xp = xp
        self.yp = yp
        self.zp = zp
        self.diffusivity = diffusivity
        self.capacity = capacity
        self.boundary = boundary.lower()

    def _instantaneous(self, x, y, z, t, tp):
        """
        Calculate temperature rise due to a 1J instant point source at time t.

        Carslaw and Jaeger page 256, 10.2(2)

        Parameters:
            x, y, z: location for desired temperature [meters]
            t: time of desired temperature [seconds]
            tp: time of source impulse [seconds]

        Returns:
            normalized temperature
        """
        if t <= tp:
            return 0

        r = np.sqrt((x - self.xp)**2 + (y - self.yp)**2 + (z - self.zp)**2)
        factor = self.capacity * 8 * (np.pi * self.diffusivity * (t - tp))**1.5
        T = 1 / factor * np.exp(-r**2 / (4 * self.diffusivity * (t - tp)))

        if self.boundary != 'infinite':
            r = np.sqrt((x - self.xp)**2 + (y - self.yp)**2 + (z + self.zp)**2)
            factor = self.capacity * 8 * (np.pi * self.diffusivity * (t - tp))**1.5
            T1 = 1 / factor * np.exp(-r**2 / (4 * self.diffusivity * (t - tp)))

            if self.boundary == 'adiabatic':
                T += T1

            if self.boundary == 'constant':
                T -= T1

        return T

    def instantaneous(self, x, y, z, t, tp):
        """
        Calculate temperature rise due to a 1J instant point source at time(s) t.

        Parameters:
            x, y, z: location for desired temperature [meters]
            t: time(s) of desired temperature [seconds]
            tp: time(s) of source impulse [seconds]

        Returns:
            Temperature Increase [°C]
        """
        T = 0
        if np.isscalar(t):
            if np.isscalar(tp):
                T = self._instantaneous(x, y, z, t, tp)
            else:
                T = np.empty_like(tp)
                for i, tt in enumerate(tp):
                    T[i] = self._instantaneous(x, y, z, t, tt)
        else:
            if np.isscalar(tp):
                T = np.empty_like(t)
                for i, tt in enumerate(t):
                    T[i] = self._instantaneous(x, y, z, tt, tp)
            else:
                raise ValueError('One of t or tp must be a scalar.')
        return T

    def _continuous(self, x, y, z, t):
        """
        Calculate temperature rise of a 1W continuous point source at time t.

        Carslaw and Jaeger page 261, 10.4(2)

        Parameters:
            x, y, z: location for desired temperature [meters]
            t: time of desired temperature [seconds]

        Returns:
            Temperature Increase [°C]
        """
        if t <= 0:
            return 0

        r = np.sqrt((x - self.xp)**2 + (y - self.yp)**2 + (z - self.zp)**2)
        factor = 1 / self.capacity / (4 * np.pi * self.diffusivity * r)
        T = factor * scipy.special.erfc(r / np.sqrt(4 * self.diffusivity * t))

        if self.boundary != 'infinite':
            r = np.sqrt((x - self.xp)**2 + (y - self.yp)**2 + (z + self.zp)**2)
            factor = 1 / self.capacity / (4 * np.pi * self.diffusivity * r)
            T1 = factor * scipy.special.erfc(r / np.sqrt(4 * self.diffusivity * t))

            if self.boundary == 'adiabatic':
                T += T1

            if self.boundary == 'constant':
                T -= T1

        return T

    def continuous(self, x, y, z, t):
        """
        Calculate temperature rise of a 1W continuous point source.

        The point source turns on at t=0.

        Parameters:
            x, y, z: location for desired temperature [meters]
            t: time(s) of desired temperature [seconds]

        Returns:
            Temperature Increase [°C]
        """
        if np.isscalar(t):
            T = self._continuous(x, y, z, t)

        else:
            T = np.empty_like(t)
            for i, tt in enumerate(t):
                T[i] = self._continuous(x, y, z, tt)
        return T

    def _pulsed(self, x, y, z, t, t_pulse):
        """
        Calculate temperature rise due to a 1J pulsed point source at time(s) t.

        1J of heat deposited at (xp, yp, zp) from t=0 to t=t_pulse.

        Parameters:
            x, y, z: location for desired temperature [meters]
            t: time of desired temperature [seconds]
            t_pulse: duration of pulse [seconds]

        Returns:
            Temperature Increase [°C]
        """
        if t <= 0:
            T = 0
        else:
            T = self._continuous(x, y, z, t)
            if t > t_pulse:
                T -= self._continuous(x, y, z, t - t_pulse)
        return T / t_pulse

    def pulsed(self, x, y, z, t, t_pulse):
        """
        Calculate temperature rise due to a 1J pulsed point source.

        Parameters:
            x, y, z: location for desired temperature [meters]
            t: time(s) of desired temperature [seconds]
            t_pulse: duration of pulse [seconds]

        Returns
            Temperature Increase [°C]
        """
        if np.isscalar(t):
            T = self._pulsed(x, y, z, t, t_pulse)
        else:
            T = np.empty_like(t)
            for i, tt in enumerate(t):
                T[i] = self._pulsed(x, y, z, tt, t_pulse)
        return T
