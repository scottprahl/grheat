# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=consider-using-f-string
# pylint: disable=no-member
# pylint: disable=too-many-instance-attributes
"""
Green's function heat transfer solutions for a vertical exponential line.

This module provides solutions to heat transfer for point illumination
of an absorbing semi-infinite medium. The solutions are
based on the mathematical formulations provided in Carslaw and Jaeger's work.

The `ExpSource` class represents a vertical linear heat source that extends
downward in the medium for coordinates (xp, yp) in the medium. The medium's surface
is defined by z=0. The class provides methods to calculate the temperature rise
at any position (x, y, z) at a specified time `t`, due to different types of
heat source behaviors.

Three types of line sources are supported:

- **Instantaneous**:
  Represents a single, instantaneous illumination of a point on absorbing medium.

- **Continuous**:
  Represents a continuous point illumination of an absorbing medium from t=0.

- **Pulsed**:
  Represents a pulsed illumination of a point from t=0 to t=`t_pulse`.

Each of these line sources can be analyzed under different boundary conditions at z=0:

- `'infinite'`: No boundary (infinite medium).
- `'adiabatic'`: No heat flow across the boundary.
- `'zero'`: Boundary is fixed at T=0.

More documentation at <https://grheat.readthedocs.io>
"""

import numpy as np
import grheat

water_heat_capacity = 4.184 * 1e6           # J/degree / m**3
water_thermal_diffusivity = 0.14558 * 1e-6  # m**2/s


class ExpSource:
    """
    Green's function heat transfer solutions for point source in infinite media.
    """

    def __init__(self,
                 mu_a,
                 xp, yp,
                 diffusivity=water_thermal_diffusivity,
                 capacity=water_heat_capacity,
                 boundary='infinite',
                 n_quad=100):
        """
        Initialize ExpSource object.

        Args:
            mu_a: attenuation coefficient                  [1/meter]
            xp: x location of source                     [meters]
            yp: y location of source                     [meters]
            tp (scalar): Time at which the source impulse occurs [seconds].
            diffusivity: thermal diffusivity             [m**2/s]
            capacity: volumetric heat capacity           [J/degree/m**3]
            boundary: 'infinite', 'adiabatic', 'zero'
            n_quad: number of quadrature points
        """
        self.mu_a = mu_a
        self.xp = xp
        self.yp = yp
        self.diffusivity = diffusivity
        self.capacity = capacity
        self.boundary = boundary.lower()

        # place source points for exponential quadrature
        k = np.arange(1, n_quad + 1)
        zz = (1 / mu_a) * np.log(2 * k)
        self.point = grheat.Point(xp, yp, zz)
        self.weights = (1 / mu_a) * (1 / (2 * k - 1) - 1 / (2 * k + 1))

    def instantaneous(self, x, y, z, t):
        """
        Calculate temperature rise due to a 1J instant point source at time(s) t.

        Args:
            x, y, z: location for desired temperature [meters]
            t: time(s) of desired temperature [seconds]
            tp: time(s) of source impulse [seconds]

        Returns:
            temperature Increase [°C]
        """
        # the contribution from each point source self.zp
        integrand = self.point.instantaneous(x, y, z, t) * np.exp(-self.mu_a * self.zp)

        # Compute the weighted sum to approximate the integral
        T = np.sum(self.weights * integrand) * self.mu_a / self.capacity

        return T

    def continuous(self, x, y, z, t):
        """
        Calculate temperature rise due to a 1W pulsed point illumination of surface.

        The point source turns on at t=0.

        Args:
            x (scalar or array): x-coord(s) for temperature calculation [meters].
            y (scalar or array): y-coord(s) for temperature calculation [meters].
            z (scalar or array): z-coord(s) for temperature calculation [meters].
            t (scalar or array): Time(s) at which the temperature is desired [seconds].

        Returns:
            temperature Increase [°C]
        """
        # the contribution from each point source self.zp
        integrand = self.point.continuous(x, y, z, t) * np.exp(-self.mu_a * self.zp)

        # Compute the weighted sum to approximate the integral
        T = np.sum(self.weights * integrand) * self.mu_a / self.capacity

        return T

    def pulsed(self, x, y, z, t, t_pulse):
        """
        Calculate temperature rise due to a 1J pulsed point illumination of surface.

        Args:
            x (scalar or array): x-coord(s) for temperature calculation [meters].
            y (scalar or array): y-coord(s) for temperature calculation [meters].
            z (scalar or array): z-coord(s) for temperature calculation [meters].
            t (scalar or array): Time(s) at which the temperature is desired [seconds].
            t_pulse (scalar): Duration of the irradiance pulse [seconds].

        Returns
            temperature Increase [°C]

        Typical usage::

            import grheat
            import numpy as np
            import matplotlib.pyplot as plt

            t = np.linspace(0, 500, 100) / 1000   # seconds
            mu_a = 0.25 * 1000                      # 1/m
            x, y, z = 0, 0, 0                         # meters
            xp,yp = 0, 0                           # meters
            t_pulse = 0.100                       # seconds

            medium = grheat.ExpSource(mu_a, xp, yp)
            T = medium.pulsed(x, y, z, t, t_pulse)

            plt.plot(t * 1000, T, color='blue')
            plt.xlabel("Time (ms)")
            plt.ylabel("temperature Increase (°C)")
            plt.title("1J pulse lasting %.0f ms" % t_pulse)
            plt.show()

        """
        # the contribution from each point source self.zp
        integrand = self.point.pulsed(x, y, z, t, t_pulse) * np.exp(-self.mu_a * self.zp)

        # Compute the weighted sum to approximate the integral
        T = np.sum(self.weights * integrand) * self.mu_a / self.capacity

        return T
