# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=consider-using-f-string
# pylint: disable=no-member
# pylint: disable=too-many-instance-attributes
"""
Green's function solutions for point illumination of semi-infinite absorber.

This module provides solutions to heat transfer for point illumination of an
absorbing semi-infinite medium. The solutions are based on the mathematical
formulations provided in Carslaw and Jaeger's work.

The `AbsorbingPoint` class represents a vertical linear exponentially-decaying,
heat source that extends downward in the medium for coordinates (xp, yp) in the
medium. The medium's surface is defined by z=0. The class provides methods to
calculate the temperature rise at any position (x, y, z) at a specified time
`t`, due to different types of heat source behaviors.

Calculations are done using exponential quadrature along all source points.

Three types of line sources are supported:

- **Instantaneous**:
  Instantaneous pulse of light at a point on semi-infinite absorber.

- **Continuous**:
  Continuous illumination of a point on semi-infinite absorber.

- **Pulsed**:
  Pulsed illumination of a point on semi-infinite absorber.

Each of these line sources can be analyzed under different boundary conditions at z=0:

- `'infinite'`: No boundary (infinite medium).
- `'adiabatic'`: No heat flow across the boundary.
- `'zero'`: Boundary is fixed at T=0.

More documentation at <https://grheat.readthedocs.io>
"""

import numpy as np
import grheat

water_heat_capacity = 4.184 * 1e6           # J/(m³ °C)
water_thermal_diffusivity = 0.14558 * 1e-6  # m²/s


class AbsorbingPoint:
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
        Initialize AbsorbingPoint object.

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
        self.tp = 0
        self.diffusivity = diffusivity
        self.capacity = capacity
        self.boundary = boundary.lower()

        # place source points for exponential quadrature
        k = np.arange(1, n_quad + 1)
        self.zp = (1 / mu_a) * np.log(2 * k)
        self.point = grheat.Point(xp, yp, self.zp)
        self.weights = (1 / mu_a) * (1 / (2 * k - 1) - 1 / (2 * k + 1))

    def instantaneous(self, x, y, z, t):
        """
        Calculate temperature rise due to a 1J instant point source at time(s) t.

        Args:
            x (scalar): x-coord(s) for temperature calculation [meters].
            y (scalar): y-coord(s) for temperature calculation [meters].
            z (scalar): z-coord(s) for temperature calculation [meters].
            t (scalar or array): Time(s) at which the temperature is desired [seconds].

        Returns:
            temperature Increase [°C]
        """
        tt = np.asarray(t)
        T = np.zeros_like(tt, dtype=float)

        for w, zp in zip(self.weights, self.zp):
            # the contribution from each point source self.zp
            self.point.zp = zp
            integrand = self.point.instantaneous(x, y, z, t) * np.exp(-self.mu_a * zp)
            T += w * integrand

        return T * self.mu_a / self.capacity

    def continuous(self, x, y, z, t):
        """
        Calculate temperature rise due to a 1W pulsed point illumination of surface.

        The point source turns on at t=0.

        Args:
            x (scalar): x-coord(s) for temperature calculation [meters].
            y (scalar): y-coord(s) for temperature calculation [meters].
            z (scalar): z-coord(s) for temperature calculation [meters].
            t (scalar or array): Time(s) at which the temperature is desired [seconds].

        Returns:
            temperature Increase [°C]
        """
        tt = np.asarray(t)
        T = np.zeros_like(tt, dtype=float)

        for w, zp in zip(self.weights, self.zp):
            # the contribution from each point source self.zp
            self.point.zp = zp
            integrand = self.point.continuous(x, y, z, t) * np.exp(-self.mu_a * zp)
            T += w * integrand

        return T * self.mu_a / self.capacity

    def pulsed(self, x, y, z, t, t_pulse):
        """
        Calculate temperature rise due to a 1J pulsed point illumination of surface.

        Args:
            x (scalar): x-coord(s) for temperature calculation [meters].
            y (scalar): y-coord(s) for temperature calculation [meters].
            z (scalar): z-coord(s) for temperature calculation [meters].
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

            medium = grheat.AbsorbingPoint(mu_a, xp, yp)
            T = medium.pulsed(x, y, z, t, t_pulse)

            plt.plot(t * 1000, T, color='blue')
            plt.xlabel("Time (ms)")
            plt.ylabel("temperature Increase (°C)")
            plt.title("1J pulse lasting %.0f ms" % t_pulse)
            plt.show()

        """
        tt = np.asarray(t)
        T = np.zeros_like(tt, dtype=float)

        for w, zp in zip(self.weights, self.zp):
            # the contribution from each point source self.zp
            self.point.zp = zp
            integrand = self.point.pulsed(x, y, z, t, t_pulse) * np.exp(-self.mu_a * zp)
            T += w * integrand

        return T * self.mu_a / self.capacity
