# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=consider-using-f-string
# pylint: disable=no-member
"""
Green's function heat transfer solutions for exponential source in infinite media.

This module provides solutions to heat transfer for uniform
illumination of am absorbing semi-infinite medium. The solutions are
based on the mathematical formulations provided in Carslaw and Jaeger's work.

The `Line` class represents a linear heat source that extends along all x-values passing 
through coordinates (yp, zp) in the medium. The medium's surface is defined by z=0. The 
class provides methods to calculate the temperature rise at any position (y, z) at a 
specified time `t`, due to different types of heat source behaviors.

Three types of line sources are supported:

- **Instantaneous**: 
  Represents a single, instantaneous release of heat along the x-line at time `tp`.

- **Continuous**: 
  Represents a continuous release of heat along the x-line source starting at t=0.

- **Pulsed**: 
  Represents a pulsed release of heat along the line source from t=0 to t=`t_pulse`.

Each of these line sources can be analyzed under different boundary conditions at z=0:

- `'infinite'`: No boundary (infinite medium).

- `'adiabatic'`: No heat flow across the boundary.

- `'zero'`: Boundary is fixed at T=0.

Any other boundary condition will trigger a ValueError.

More documentation can be found at `grheat Documentation <https://grheat.readthedocs.io>`_.

"""

import scipy.special
import numpy as np

water_heat_capacity = 4.184 * 1e6           # J/degree / m**3
water_thermal_diffusivity = 0.14558 * 1e-6  # m**2/s

class ExpSource:
    """
    Green's function heat transfer solutions for point source in infinite media.

    """

    def __init__(self,
                 mu,
                 yp, zp,
                 diffusivity=water_thermal_diffusivity,
                 capacity=water_heat_capacity,
                 boundary='infinite'):
        """
        Initialize ExpSource object.

        Parameters:
            mu: attenuation coefficient                  [1/meter]
            xp: x location of source                     [meters]
            yp: y location of source                     [meters]
            diffusivity: thermal diffusivity             [m**2/s]
            capacity: volumetric heat capacity           [J/degree/m**3]
            boundary: 'infinite', 'adiabatic', 'zero'
        Returns:
            Planar heat source object
        """
        self.mu = mu
        self.xp = xp
        self.yp = yp
        self.diffusivity = diffusivity
        self.capacity = capacity
        self.boundary = boundary.lower()
        self.point = grheat.Point(xp, yp, 0)

    def _instantaneous(self, x, y, z, t, tp):
        """
        Calculate temperature rise due to a 1J instant exp source at time t.

        T = int( exp(-mu * zp)*G(x-xp,y-yp,z-zp) dzp

        Parameters:
            x, y, z: location for desired temperature [meters]
            t: time of desired temperature [seconds]
            tp: time of source impulse [seconds]

        Returns:
            normalized temperature
        """
        T = 0
        if t > tp:

            # integrate from t to tp over all zp
            for zpp in np.linspace(0,5):
                self.point.zp = zpp
                T += self.point._instantaneous(x,y,z,t,tp) * exp(-mu*zp)

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
        T = 0
        if t > tp:

            # integrate from t to tp over all zp
            for zpp in np.linspace(0,5):
                self.point.zp = zpp
                T += self.point._continuous(x,y,z,t,tp) * exp(-mu*zp)

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

        Typical usage::

            import grheat
            import numpy as np
            import matplotlib.pyplot as plt

            t = np.linspace(0, 500, 100) / 1000   # seconds
            mu = 0.25 * 1000                      # 1/m
            x,y,z = 0,0,0                         # meters
            xp,yp = 0,0                           # meters
            t_pulse = 0.100                       # seconds

            medium = grheat.ExpSource(mu, xp, yp)
            T = medium.pulsed(x,y,z,t,t_pulse)

            plt.plot(t * 1000, T, color='blue')
            plt.xlabel("Time (ms)")
            plt.ylabel("Temperature Increase (°C)")
            plt.title("1J pulse lasting %.0f ms" % t_pulse)
            plt.show()

        """
        if np.isscalar(t):
            T = self._pulsed(x, y, z, t, t_pulse)
        else:
            T = np.empty_like(t)
            for i, tt in enumerate(t):
                T[i] = self._pulsed(x, y, z, tt, t_pulse)
        return T
