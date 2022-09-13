# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=consider-using-f-string
# pylint: disable=no-member
"""
Green's function heat transfer solutions for x-line source in semi-infinite media.

The surface is defined by z=0 and the line source extends horizontally
from -∞ < z < +∞.

More documentation at <https://grheat.readthedocs.io>

    Typical usage::

        import grheat
        import numpy as np
        import matplotlib.pyplot as plt

        t = np.linspace(0, 500, 100) / 1000   # seconds
        y, z = 0,0                            # meters
        yp, zp = 0,0.001                      # meters
        t_pulse = 0.100                       # seconds

        line = grheat.Line(yp, zp)
        T = line.pulsed(y, z, t, t_pulse)

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
        y, z = 0,0                            # meters
        yp, zp = 0,0.001                      # meters
        t_pulse = 0.100                       # seconds

        line = grheat.Line(yp, zp)
        T = line.pulsed(y, z, t, t_pulse)

        plt.plot(t * 1000, T, color='blue')
        plt.xlabel("Time (ms)")
        plt.ylabel("Temperature Increase (°C)")
        plt.title("1J pulse lasting %.0f ms" % t_pulse)
        plt.show()
    """

    def __init__(self,
                 zp,
                 yp,
                 diffusivity=water_thermal_diffusivity,
                 capacity=water_heat_capacity,
                 boundary='infinite'):
        """
        Initialize Line object.

        Parameters:
            yp: y location of x-line                     [meters]
            zp: z location of x-line                     [meters]
            diffusivity: thermal diffusivity             [m**2/s]
            capacity: heat capacity                      [J/degree/kg]
            boundary: 'infinite', 'adiabatic', 'zero'
        Returns:
            Planar heat source object
        """
        self.yp = yp
        self.zp = zp
        self.diffusivity = diffusivity     # m**2/s
        self.capacity = capacity           # J/degree/kg
        self.boundary = boundary.lower()   # infinite, adiabatic, zero

    def _instantaneous(self, y, z, t, tp):
        """
        Calculate temperature rise due to a 1 J/m instantaneous x-line source at time t.

        The line parallel to the x-axis and passes through yp, zp.

        Carslaw and Jaeger page 258, 10.3(1)

        Parameters:
            y, z: location for desired temperature [meters]
            t: time of desired temperature [seconds]
            tp: time of source impulse [seconds]

        Returns:
            normalized temperature
        """
        if t <= tp:
            return 0

        r2 = (y - self.yp)**2 + (z - self.zp)**2
        factor = self.capacity * 4 * np.pi * self.diffusivity * (t - tp)
        T = 1 / factor * np.exp(-r2 / (4 * self.diffusivity * (t - tp)))

        if self.boundary != 'infinite':
            r2 = (y - self.yp)**2 + (z + self.zp)**2
            T1 = 1 / factor * np.exp(-r2 / (4 * self.diffusivity * (t - tp)))

            if self.boundary == 'adiabatic':
                T += T1

            if self.boundary == 'zero':
                T -= T1

        return T

    def instantaneous(self, y, z, t, tp):
        """
        Calculate temperature rise due to a 1 J/m instant x-line source.

        Parameters:
            y, z: location for desired temperature [meters]
            t: time of desired temperature [seconds]
            tp: time of source impulse [seconds]

        Returns:
            Temperature Increase [°C]
        """
        if np.isscalar(t):
            T = self._instantaneous(y, z, t, tp)

        else:
            T = np.empty_like(t)
            for i, tt in enumerate(t):
                T[i] = self._instantaneous(y, z, tt, tp)
        return T

    def _continuous(self, y, z, t):
        """
        Calculate temperature rise due to a 1W/m x-line source at single time point.

        Carslaw and Jaeger page 261, 10.4(5)

        Parameters:
            y, z: location for desired temperature [meters]
            t: time of desired temperature [seconds]

        Returns:
            Temperature Increase [°C]
        """
        if t <= 0:
            return 0

        r2 = (y - self.yp)**2 + (z - self.zp)**2
        factor = -self.capacity * 4 * np.pi * self.diffusivity
        T = 1 / factor * scipy.special.expi(-r2 / (4 * self.diffusivity * t))

        if self.boundary != 'infinite':
            r2 = (y - self.yp)**2 + (z + self.zp)**2
            T1 = 1 / factor * scipy.special.expi(-r2 / (4 * self.diffusivity * t))

            if self.boundary == 'adiabatic':
                T += T1

            if self.boundary == 'zero':
                T -= T1

        return T

    def continuous(self, y, z, t):
        """
        Calculate temperature rise due to a 1W/m continuous x-line source.

        The x-line source turns on at t=0 and passes through (yp, zp).

        Parameters:
            y, z: location for desired temperature [meters]
            t: time(s) of desired temperature [seconds]

        Returns:
            Temperature Increase [°C]
        """
        if np.isscalar(t):
            T = self._continuous(y, z, t)

        else:
            T = np.empty_like(t)
            for i, tt in enumerate(t):
                T[i] = self._continuous(y, z, tt)
        return T

    def _pulsed(self, y, z, t, t_pulse):
        """
        Calculate temperature rise due to a 1 J/m pulsed x-line source at time(s) t.

        1 J/m of heat deposited along x-line passing through (yp, zp) from t=0 to t=t_pulse.

        Parameters:
            y, z: location for desired temperature [meters]
            t: time of desired temperature [seconds]
            t_pulse: duration of pulse [seconds]

        Returns:
            Temperature Increase [°C]
        """
        if t <= 0:
            T = 0
        else:
            T = self._continuous(y, z, t)
            if t > t_pulse:
                T -= self._continuous(y, z, t - t_pulse)
        return T / t_pulse

    def pulsed(self, y, z, t, t_pulse):
        """
        Calculate temperature rise due to a 1 J/m pulsed x-line source.

        Parameters:
            y, z: location for desired temperature [meters]
            t: time(s) of desired temperature [seconds]
            t_pulse: duration of pulse [seconds]

        Returns
            Temperature Increase [°C]
        """
        if np.isscalar(t):
            T = self._pulsed(y, z, t, t_pulse)
        else:
            T = np.empty_like(t)
            for i, tt in enumerate(t):
                T[i] = self._pulsed(y, z, tt, t_pulse)
        return T
