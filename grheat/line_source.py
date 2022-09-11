# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=consider-using-f-string
# pylint: disable=no-member
"""
Green's function heat transfer solutions for z-line source in infinite media.

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

        line = grheat.Line()
        T = line.pulsed(x,y,z,t,xp,yp,t_pulse)

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


class Line:
    """
    Green's function heat transfer solutions for line source in infinite media.

    Typical usage::

        import grheat
        import numpy as np
        import matplotlib.pyplot as plt

        t = np.linspace(0, 500, 100) / 1000   # seconds
        mua = 0.25 * 1000                     # 1/m
        x,y,z = 0,0,0                         # meters
        xp,yp,zp = 0,0,0.001                  # meters
        t_pulse = 0.100                       # seconds

        line = grheat.Line()
        T = line.pulsed(x,y,z,t,xp,yp,t_pulse)

        plt.plot(t * 1000, T, color='blue')
        plt.xlabel("Time (ms)")
        plt.ylabel("Temperature Increase (°C)")
        plt.title("1J pulse lasting %.0f ms" % t_pulse)
        plt.show()
    """

    def __init__(self,
                 xp,
                 yp,
                 diffusivity=water_thermal_diffusivity,
                 capacity=water_heat_capacity,
                 boundary='infinite'):
        """
        Initialize Line object.

        Parameters:
            xp: x location of z-line                     [meters]
            yp: y location of z-line                     [meters]
            diffusivity: thermal diffusivity             [m**2/s]
            capacity: heat capacity                      [J/degree/kg]
            boundary: 'infinite', 'adiabatic', 'constant'
        Returns:
            Planar heat source object
        """
        self.xp = xp
        self.yp = yp
        self.diffusivity = diffusivity     # m**2/s
        self.capacity = capacity           # J/degree/kg
        self.boundary = boundary.lower()   # infinite, adiabatic, constant

    def _instantaneous(self, x, y, t, tp):
        """
        Calculate temperature rise due to a 1 J/m instantaneous z-line source at time t.

        The line parallel to the z-axis and passes through xp, yp.

        Carslaw and Jaeger page 258, 10.3(1)

        Parameters:
            x, y: location for desired temperature [meters]
            t: time of desired temperature [seconds]
            tp: time of source impulse [seconds]

        Returns:
            normalized temperature
        """
        if t <= tp:
            return 0

        r = np.sqrt((x - self.xp)**2 + (y - self.yp)**2)
        factor = self.capacity * 4 * np.pi * self.diffusivity * (t - tp)
        return 1 / factor * np.exp(-r**2 / (4 * self.diffusivity * (t - tp)))

    def instantaneous(self, x, y, t, tp):
        """
        Calculate temperature rise due to a 1 J/m instant z-line source.

        Parameters:
            x, y: location for desired temperature [meters]
            t: time of desired temperature [seconds]
            tp: time of source impulse [seconds]

        Returns:
            Temperature Increase [°C]
        """
        if np.isscalar(t):
            T = self._instantaneous(x, y, t, tp)

        else:
            T = np.empty_like(t)
            for i, tt in enumerate(t):
                T[i] = self._instantaneous(x, y, tt, tp)
        return T

    def _continuous(self, x, y, t):
        """
        Calculate temperature rise due to a 1W/m z-line source at single time point.

        Carslaw and Jaeger page 261, 10.4(5)

        Parameters:
            x, y: location for desired temperature [meters]
            t: time of desired temperature [seconds]

        Returns:
            Temperature Increase [°C]
        """
        if t <= 0:
            return 0

        r = np.sqrt((x - self.xp)**2 + (y - self.yp)**2)
        factor = -self.capacity * 4 * np.pi * self.diffusivity
        return 1 / factor * scipy.special.expi(-r**2 / (4 * self.diffusivity * t))

    def continuous(self, x, y, t):
        """
        Calculate temperature rise due to a 1W/m continuous z-line source.

        The z-line source turns on at t=0 and passes through (xp, yp).

        Parameters:
            x, y: location for desired temperature [meters]
            t: time(s) of desired temperature [seconds]

        Returns:
            Temperature Increase [°C]
        """
        if np.isscalar(t):
            T = self._continuous(x, y, t)

        else:
            T = np.empty_like(t)
            for i, tt in enumerate(t):
                T[i] = self._continuous(x, y, tt)
        return T

    def _pulsed(self, x, y, t, t_pulse):
        """
        Calculate temperature rise due to a 1 J/m pulsed z-line source at time(s) t.

        1 J/m of heat deposited along z-line passing through (xp, yp) from t=0 to t=t_pulse.

        Parameters:
            x, y: location for desired temperature [meters]
            t: time of desired temperature [seconds]
            t_pulse: duration of pulse [seconds]

        Returns:
            Temperature Increase [°C]
        """
        if t <= 0:
            T = 0
        else:
            T = self._continuous(x, y, t)
            if t > t_pulse:
                T -= self._continuous(x, y, t - t_pulse)
        return T / t_pulse

    def pulsed(self, x, y, t, t_pulse):
        """
        Calculate temperature rise due to a 1 J/m pulsed z-line source.

        Parameters:
            x, y: location for desired temperature [meters]
            t: time(s) of desired temperature [seconds]
            t_pulse: duration of pulse [seconds]

        Returns
            Temperature Increase [°C]
        """
        if np.isscalar(t):
            T = self._pulsed(x, y, t, t_pulse)
        else:
            T = np.empty_like(t)
            for i, tt in enumerate(t):
                T[i] = self._pulsed(x, y, tt, t_pulse)
        return T
