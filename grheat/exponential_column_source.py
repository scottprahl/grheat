"""Green's function solutions for exponentially decaying vertical source columns.

This module provides solutions for heat transfer from point illumination of an
absorbing semi-infinite medium. The solutions are based on the mathematical
formulations provided in Carslaw and Jaeger.

The ``ExponentialColumnSource`` class represents a vertical column of point
sources beneath ``(xp, yp)`` whose strength decays exponentially with depth. The
medium's surface is defined by ``z = 0``. The class provides methods to
calculate the temperature rise at any position ``(x, y, z)`` at a specified
time ``t`` for several temporal source profiles.

Calculations are done using exponential quadrature along the source depth.

Supported source profiles:

- **Instantaneous**: Instantaneous pulse of light at a surface point.
- **Continuous**: Continuous illumination of a surface point.
- **Pulsed**: Finite-duration illumination of a surface point.

Each profile can be analyzed under different boundary conditions at ``z = 0``:

- ``'infinite'``: No boundary (infinite medium).
- ``'adiabatic'``: No heat flow across the boundary.
- ``'zero'``: Boundary is fixed at ``T = 0``.

More documentation at <https://grheat.readthedocs.io>
"""

import numpy as np
import grheat

water_heat_capacity = 4.184 * 1e6  # J/(m³ °C)
water_thermal_diffusivity = 0.14558 * 1e-6  # m²/s


class ExponentialColumnSource:
    """Green's function solutions for an exponentially weighted vertical source column."""

    def __init__(
        self,
        mu_a,
        xp,
        yp,
        diffusivity=water_thermal_diffusivity,
        capacity=water_heat_capacity,
        boundary="infinite",
        n_quad=100,
    ):
        """Initialize an exponentially weighted vertical source column.

        Args:
            mu_a (scalar): Exponential attenuation coefficient of the absorber [1/meters].
            xp (scalar): x-coordinate of the source-column origin [meters].
            yp (scalar): y-coordinate of the source-column origin [meters].
            diffusivity (scalar, optional): Thermal diffusivity of the medium [m^2/s].
                Defaults to ``water_thermal_diffusivity``.
            capacity (scalar, optional): Volumetric heat capacity of the medium
                [J/degree/m^3]. Defaults to ``water_heat_capacity``.
            boundary (str, optional): Boundary condition label stored on the instance.
                Defaults to ``"infinite"``.
            n_quad (int, optional): Number of quadrature points used to approximate the
                exponentially distributed source depth. Defaults to ``100``.
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
        self.point = grheat.Point(xp, yp, self.zp, boundary=self.boundary)
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
            scalar or array: Temperature increase at the requested point(s) [°C].
        """
        T = 0.0

        for w, zp in zip(self.weights, self.zp):
            # the contribution from each point source self.zp
            self.point.zp = zp
            integrand = self.point.instantaneous(x, y, z, t) * np.exp(-self.mu_a * zp)
            T = T + w * integrand

        return T * self.mu_a / self.capacity

    def continuous(self, x, y, z, t):
        """
        Calculate temperature rise due to 1 W of continuous surface point illumination.

        The point source turns on at t=0.

        Args:
            x (scalar): x-coord(s) for temperature calculation [meters].
            y (scalar): y-coord(s) for temperature calculation [meters].
            z (scalar): z-coord(s) for temperature calculation [meters].
            t (scalar or array): Time(s) at which the temperature is desired [seconds].

        Returns:
            scalar or array: Temperature increase at the requested point(s) [°C].
        """
        T = 0.0

        for w, zp in zip(self.weights, self.zp):
            # the contribution from each point source self.zp
            self.point.zp = zp
            integrand = self.point.continuous(x, y, z, t) * np.exp(-self.mu_a * zp)
            T = T + w * integrand

        return T * self.mu_a / self.capacity

    def pulsed(self, x, y, z, t, t_pulse):
        """
        Calculate temperature rise due to a finite pulse of surface point illumination.

        Args:
            x (scalar): x-coord(s) for temperature calculation [meters].
            y (scalar): y-coord(s) for temperature calculation [meters].
            z (scalar): z-coord(s) for temperature calculation [meters].
            t (scalar or array): Time(s) at which the temperature is desired [seconds].
            t_pulse (scalar): Duration of the irradiance pulse [seconds].

        Returns:
            scalar or array: Temperature increase at the requested point(s) [°C].

        Typical usage::

            import grheat
            import numpy as np
            import matplotlib.pyplot as plt

            t = np.linspace(0, 500, 100) / 1000   # seconds
            mu_a = 0.25 * 1000                      # 1/m
            x, y, z = 0, 0, 0                         # meters
            xp,yp = 0, 0                           # meters
            t_pulse = 0.100                       # seconds

            medium = grheat.ExponentialColumnSource(mu_a, xp, yp)
            T = medium.pulsed(x, y, z, t, t_pulse)

            plt.plot(t * 1000, T, color='blue')
            plt.xlabel("Time (ms)")
            plt.ylabel("temperature Increase (°C)")
            plt.title("1J pulse lasting %.0f ms" % t_pulse)
            plt.show()

        """
        T = 0.0

        for w, zp in zip(self.weights, self.zp):
            # the contribution from each point source self.zp
            self.point.zp = zp
            integrand = self.point.pulsed(x, y, z, t, t_pulse) * np.exp(-self.mu_a * zp)
            T = T + w * integrand

        return T * self.mu_a / self.capacity
