# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=consider-using-f-string
# pylint: disable=no-member
"""
x-Line Source
=============

This module provides Green's function solutions for heat transfer due to an x-line
source in a semi-infinite medium, encapsulated within the `Line` class. The solutions are
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

More documentation at <https://grheat.readthedocs.io>

"""
import scipy.special
import numpy as np

# Constants for water properties
water_heat_capacity = 4.184 * 1e6           # J/degree / m**3
water_thermal_diffusivity = 0.14558 * 1e-6  # m**2/s


class Line:
    """Provides Green's function heat transfer solutions for a line source.

    The Line class encapsulates the behavior of a line source situated in a
    semi-infinite medium with a surface defined at z=0. The line source extends
    horizontally along all x-values passing through coordinates (yp, zp). At time tp,
    the line source delivers a heat impulse of 1 Joule per meter along its length.

    Boundary conditions at z=0 can be:
        - 'infinite': No boundary (infinite medium).
        - 'adiabatic': No heat flow across the boundary.
        - 'zero': Boundary is fixed at T=0.

    Attributes:
        yp (scalar): y-coordinate of the line source. [meters]
        zp (scalar): z-coordinate of the line source. [meters]
        diffusivity (scalar): Thermal diffusivity of the medium. [m^2/s]
        capacity (scalar): Volumetric heat capacity of the medium. [J/degree/m^3]
        boundary (str): Boundary condition at z=0. ['infinite', 'adiabatic', 'zero']

    """

    def __init__(self,
                 yp, zp,
                 tp=0,
                 diffusivity=water_thermal_diffusivity,
                 capacity=water_heat_capacity,
                 boundary='infinite'):
        """
        Initializes a Line object representing a line source in a medium.

        The line source extends infinitely parallel to the x-axis and passes through
        the coordinates (yp, zp) in the medium. At time tp, the line source delivers a
        heat impulse of 1 Joule per meter along the line.

        The surface of the medium is defined by z=0 and the boundary conditions may be:
            - 'infinite': No boundary.
            - 'adiabatic': No heat flow across the boundary.
            - 'zero': Boundary temperature is fixed at T=0.

        Args:
            yp (scalar): The y-coordinate through which the x-line source passes. [meters]
            zp (scalar): The z-coordinate through which the x-line source passes,
                        defining its depth below the surface z=0. [meters]
            tp (scalar, optional): The time at which the line source impulse occurs. [seconds]
            diffusivity (scalar, optional): The thermal diffusivity of the medium.
                                 Defaults to water_thermal_diffusivity. [m^2/s]
            capacity (scalar, optional): The volumetric heat capacity of the medium.
                              Defaults to water_heat_capacity. [J/degree/m^3]
            boundary (str, optional): string describing boundary conditions at z=0

        Raises:
            ValueError: If the specified boundary condition is not one of 'infinite',
                        'adiabatic', or 'zero'.
        """
        self.yp = yp
        self.zp = zp
        self.tp = tp
        self.diffusivity = diffusivity     # m^2/s
        self.capacity = capacity           # J/degree/m^3
        self.boundary = boundary.lower()   # infinite, adiabatic, zero

        if self.boundary not in ['infinite', 'adiabatic', 'zero']:
            raise ValueError("boundary must be 'infinite', 'adiabatic', or 'zero'")

    def _instantaneous(self, y, z, t, tp):
        """
        Calculates temperature rise due to a 1 J/m instantaneous x-line source at time t.

        This method computes the temperature rise at a specified location and time due to
        an instantaneous x-line source with a heat impulse of 1 J/m at time `tp`. The line
        source is parallel to the x-axis and passes through the coordinates (yp, zp).

        The parameters `t` and `tp` should be scalars. If `t` is less than or equal to `tp`,
        the method returns 0 or an array of zeros depending on y and z.

        Reference: Carslaw and Jaeger (1959), page 258, Equation 10.3(1).

        Args:
            y (scalar or array): The y-coordinate(s) for the temperature location. [meters]
            z (scalar or array): The z-coordinate(s) for the temperature location. [meters]
            t (scalar): The time of the temperature. [seconds]
            tp (scalar): The time of the line source impulse. [seconds]
        Returns:
            scalar: Normalized temperature rise.
        """
        r2 = (y - self.yp)**2 + (z - self.zp)**2

        if t <= tp:
            return 0 * r2

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

    def instantaneous(self, y, z, t):
        """
        Calculates the temperature rise due to a 1 J/m instant x-line source.

        This method computes the temperature rise at a specified location and time
        due to an instant x-line source. The line source is parallel to the x-axis
        and passes through the coordinates (yp, zp).

        Args:
            y (scalar or array): The y-coordinate(s) for the temperature location. [meters]
            z (scalar or array): The z-coordinate(s) for the temperature location. [meters]
            t (scalar or array): The time(s) of the temperature. [seconds]

        Returns:
            scalar: Temperature increase in degrees Celsius [°C].

        Example:
            The following example demonstrates how to use this method to calculate and
            plot the temperature rise over time due to an instant line source:

            .. code-block:: python

                import grheat
                import numpy as np
                import matplotlib.pyplot as plt

                t = np.linspace(0, 500, 100) / 1000   # seconds
                y, z = 0, 0                           # meters
                yp, zp = 0, 0.001                     # meters

                line = grheat.Line(yp, zp)
                T = line.instantaneous(y, z, t)

                plt.plot(t * 1000, T, color='blue')
                plt.xlabel("Time (ms)")
                plt.ylabel("Surface Temperature Increase (°C)")
                plt.title("Instantaneous Line Source at 1mm depth")
                plt.show()
        """
        if np.isscalar(t):
            T = 0                # return a scalar
            if np.isscalar(self.tp):
                T += self._instantaneous(y, z, t, self.tp)
            else:
                for i, tp in enumerate(self.tp):
                    T += self._instantaneous(y, z, t, tp)

        else:
            T = np.zeros_like(t)  # return an array
            for i, tt in enumerate(t):
                if np.isscalar(self.tp):
                    T[i] += self._instantaneous(y, z, tt, self.tp)
                else:
                    for tp in self.tp:
                        T[i] += self._instantaneous(y, z, tt, tp)
        return T

    def _continuous(self, y, z, t):
        """
        Calculate temperature rise due to a 1W/m x-line source at single time point.

        Carslaw and Jaeger page 261, 10.4(5)

        Args:
            y (scalar or array): The y-coordinate(s) for the temperature location. [meters]
            z (scalar or array): The z-coordinate(s) for the temperature location. [meters]
            t (scalar): The time of the temperature. [seconds]

        Returns:
            Temperature Increase [°C]
        """
        r2 = (y - self.yp)**2 + (z - self.zp)**2

        if t <= 0:
            return 0 * r2

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
        Calculates temperature rise due to a 1W/m continuous x-line source.

        The x-line source turns on at t=0 and passes through the coordinates (yp, zp).

        Reference: Carslaw and Jaeger (1959), page 261, Equation 10.4(5).

        Parameters:
            y (scalar): The y-coordinate for the desired temperature location. [meters]
            z (scalar): The z-coordinate for the desired temperature location. [meters]
            t (scalar or array): Time(s) of desired temperature. [seconds]

        Returns:
            scalar or array: Temperature increase in degrees Celsius [°C].

        Example:
            The following example demonstrates how to use this method to calculate and
            plot the temperature rise over time due to a continuous line source 1mm
            deep that turned on at t=0:

            .. code-block:: python

                import grheat
                import numpy as np
                import matplotlib.pyplot as plt

                t = np.linspace(0, 500, 100) / 1000   # seconds
                y, z = 0, 0                           # meters
                yp, zp = 0, 0.001                     # meters

                line = grheat.Line(yp, zp)
                T = line.continuous(y, z, t)

                plt.plot(t * 1000, T, color='blue')
                plt.xlabel("Time (ms)")
                plt.ylabel("Surface Temperature Increase (°C)")
                plt.title("Continuous Line Source at 1mm depth")
                plt.show()
        """
        if np.isscalar(t):
            T = 0                # return a scalar
            if np.isscalar(self.tp):
                T += self._continuous(y, z, t - self.tp)
            else:
                for i, tp in enumerate(self.tp):
                    T += self._continuous(y, z, t - tp)

        else:
            T = np.zeros_like(t)  # return an array
            for i, tt in enumerate(t):
                if np.isscalar(self.tp):
                    T[i] += self._continuous(y, z, tt - self.tp)
                else:
                    for tp in self.tp:
                        T[i] += self._continuous(y, z, tt - tp)
        return T

    def pulsed(self, y, z, t, t_pulse):
        """
        Calculates temperature rise due to a 1 J/m pulsed x-line source.

        This method computes the temperature rise at a specified location and time due
        to a pulsed x-line source. 1 J/m of heat is deposited along the x-line passing
        through the coordinates (yp, zp) from t=0 to t=t_pulse.

        Parameters:
            y (scalar or array): The y-coordinate for the desired temperature location. [meters]
            z (scalar or array): The z-coordinate for the desired temperature location. [meters]
            t (scalar or array): Time(s) of desired temperature. [seconds]
            t_pulse (scalar): Duration of the pulse. [seconds]

        Returns:
            scalar or array: Temperature increase in degrees Celsius [°C].

        Example:
            The following example demonstrates how to use this method to calculate and
            plot the temperature rise over time due to a pulsed line source 1mm deep:

            .. code-block:: python

                import grheat
                import numpy as np
                import matplotlib.pyplot as plt

                t = np.linspace(0, 500, 100) / 1000   # seconds
                y, z = 0, 0                           # meters
                yp, zp = 0, 0.001                     # meters
                t_pulse = 0.3

                line = grheat.Line(yp, zp)
                T = line.pulsed(y, z, t, t_pulse)

                plt.plot(t * 1000, T, color='blue')
                plt.xlabel("Time (ms)")
                plt.ylabel("Surface Temperature Increase (°C)")
                plt.title("Pulsed Line Source at 1mm depth")
                plt.show()
        """
        if t_pulse < 0:
            raise ValueError("Pulse duration (%f) must be positive" % t_pulse)

        T = self.continuous(y, z, t)
        T -= self.continuous(y, z, t - t_pulse)
        return T / t_pulse
