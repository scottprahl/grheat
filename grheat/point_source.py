# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=consider-using-f-string
# pylint: disable=no-member
"""
Green's Function Heat Transfer Solutions for Point Source in Infinite Media
===========================================================================

This module provides Green's function solutions for heat transfer due to a point
source in an infinite medium, encapsulated within the `Point` class. The
solutions are based on the mathematical formulations provided in Carslaw and
Jaeger's work.

The `Point` class represents a point heat source located at a specified position
`(xp, yp, zp)` in the medium. It provides methods to calculate the temperature
rise at any given location `(x, y, z)` at a specified time `t` due to different
types of heat source behavior: instantaneous, continuous, or pulsed.

The module supports various boundary conditions such as infinite, adiabatic, or
zero boundary, and allows for specifying thermal properties like diffusivity and
volumetric heat capacity of the medium.

For further documentation and examples, visit `grheat Documentation <https://grheat.readthedocs.io>`_.
"""
import scipy.special
import numpy as np

water_heat_capacity = 4.184 * 1e6           # J/degree / m**3
water_thermal_diffusivity = 0.14558 * 1e-6  # m**2/s


class Point:
    """
    Green's function heat transfer solutions for point source in semi-infinite media.

    The `Point` class represents a point heat source located at a specified position
    `(xp, yp, zp)` in the medium. It provides methods to calculate the temperature
    rise at any given location `(x, y, z)` at a specified time `t` due to different
    types of heat source behavior: instantaneous, continuous, or pulsed.

    In addition, three types of boundary conditions are supported: infinite,
    adiabatic (for z=0), and zero (again for z=0).
    """

    def __init__(self,
                 xp, yp, zp,
                 diffusivity=water_thermal_diffusivity,
                 capacity=water_heat_capacity,
                 boundary='infinite'):
        """
        Initializes a Point object representing a point heat source in a medium.

        This method sets up a point heat source at a specified location, with specified
        thermal properties and boundary conditions.

        Args:
            xp (float): The x-coordinate of the point source in meters.
            yp (float): The y-coordinate of the point source in meters.
            zp (float): The z-coordinate of the point source in meters.
            diffusivity (float, optional): The thermal diffusivity of the medium in
                square meters per second (m**2/s). Defaults to water_thermal_diffusivity.
            capacity (float, optional): The volumetric heat capacity of the medium in
                joules per cubic meter per degree Celsius (J/degree/m**3).
                Defaults to water_heat_capacity.
            boundary (str, optional): The boundary condition applied at the surface of
                the medium. Acceptable values are 'infinite', 'adiabatic', or 'zero'.
                Defaults to 'infinite'.

        Raises:
            ValueError: If the provided boundary condition is not one of the accepted values.
        """
        self.xp = xp
        self.yp = yp
        self.zp = zp
        self.diffusivity = diffusivity
        self.capacity = capacity
        self.boundary = boundary.lower()
        if self.boundary not in ['infinite', 'adiabatic', 'zero']:
            raise ValueError("boundary must be 'infinite', 'adiabatic', or 'zero'")

    def _instantaneous(self, x, y, z, t, tp):
        """
        Calculate temperature rise due to a 1J instant point source at a specific time `t`.

        This method computes the temperature rise at a given location `(x, y, z)` at time `t`
        due to a 1J instantaneous point source occurring at time `tp`, following the formula
        from Carslaw and Jaeger (page 256, 10.2(2)). Both `t` and `tp` are assumed to be scalars.

        Args:
            x (array-like): x-coordinate(s) of the location(s) for desired temperature [meters].
            y (array-like): y-coordinate(s) of the location(s) for desired temperature [meters].
            z (array-like): z-coordinate(s) of the location(s) for desired temperature [meters].
            t (scalar): Time at which the desired temperature is computed [seconds].
            tp (scalar): Time at which the source impulse occurs [seconds].

        Returns:
            numpy.ndarray: Normalized temperature rise at the specified location(s) and time.
        """
        r = np.sqrt((x - self.xp)**2 + (y - self.yp)**2 + (z - self.zp)**2)
        if t <= tp:
            if np.isscalar(r):
                return 0
            else:
                return np.zeros_like(r)

        factor = self.capacity * 8 * (np.pi * self.diffusivity * (t - tp))**1.5
        T = 1 / factor * np.exp(-r**2 / (4 * self.diffusivity * (t - tp)))

        if self.boundary != 'infinite':
            r = np.sqrt((x - self.xp)**2 + (y - self.yp)**2 + (z + self.zp)**2)
            factor = self.capacity * 8 * (np.pi * self.diffusivity * (t - tp))**1.5
            T1 = 1 / factor * np.exp(-r**2 / (4 * self.diffusivity * (t - tp)))

            if self.boundary == 'adiabatic':
                T += T1

            if self.boundary == 'zero':
                T -= T1

        return T

    def instantaneous(self, x, y, z, t, tp):
        """
        Calculates the temperature rise due to a 1J instant point source at specified time(s).

        This method computes the temperature rise at given location(s) `(x, y, z)` at time(s) `t`
        due to a 1J instantaneous point source located at `(xp, yp, zp)` occurring at time(s) `tp`,
        following the formula from Carslaw and Jaeger (page 256, 10.2(2)). Either `t` or `tp`
        should be a scalar while the other can be an array.

        Args:
            x (array-like): x-coordinate(s) of the location(s) for desired temperature [meters].
            y (array-like): y-coordinate(s) of the location(s) for desired temperature [meters].
            z (array-like): z-coordinate(s) of the location(s) for desired temperature [meters].
            t (scalar or array-like): Time(s) at which the desired temperature is computed [seconds].
            tp (scalar or array-like): Time(s) at which the source impulse occurs [seconds].

        Raises:
            ValueError: If both `t` and `tp` are arrays. One of them must be a scalar.

        Returns:
            numpy.ndarray: Temperature increase at the specified location(s) and time(s) in degrees Celsius.

        Example:
            .. code-block:: python

                import grheat
                import numpy as np
                import matplotlib.pyplot as plt

                # Define time, source, and observation points
                t = np.linspace(0, 500, 100) / 1000   # seconds
                x, y, z = 0, 0, 0                     # meters
                xp, yp, zp = 0, 0, 0.001              # meters
                t_pulse = 0.100                       # seconds

                # Create a Point object representing the heat source
                medium = grheat.Point(xp, yp, zp)

                # Calculate the temperature rise due to a pulsed heat source
                T = medium.pulsed(x, y, z, t, t_pulse)

                # Plot the temperature rise over time
                plt.plot(t * 1000, T, color='blue')
                plt.xlabel("Time (ms)")
                plt.ylabel("Temperature Increase (°C)")
                plt.title("1J pulse lasting %.0f ms" % t_pulse)
                plt.show()
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
        r = np.sqrt((x - self.xp)**2 + (y - self.yp)**2 + (z - self.zp)**2)
        if t <= 0:
            if np.isscalar(r):
                return 0
            else:
                return np.zeros_like(r)

        factor = 1 / self.capacity / (4 * np.pi * self.diffusivity * r)
        T = factor * scipy.special.erfc(r / np.sqrt(4 * self.diffusivity * t))

        if self.boundary != 'infinite':
            r = np.sqrt((x - self.xp)**2 + (y - self.yp)**2 + (z + self.zp)**2)
            factor = 1 / self.capacity / (4 * np.pi * self.diffusivity * r)
            T1 = factor * scipy.special.erfc(r / np.sqrt(4 * self.diffusivity * t))

            if self.boundary == 'adiabatic':
                T += T1

            if self.boundary == 'zero':
                T -= T1

        return T

    def continuous(self, x, y, z, t):
        """
        Calculates the temperature rise due to a 1W continuous point source at specified time(s).

        This method computes the temperature rise at given location(s) `(x, y, z)` at time(s) `t`
        due to a 1W continuous point source located at `(xp, yp, zp)` that was turned on at time `t=0`,
        following the formula from Carslaw and Jaeger (page 261, 10.4(2)).

        Args:
            x (array-like): x-coordinate(s) of the location(s) for desired temperature [meters].
            y (array-like): y-coordinate(s) of the location(s) for desired temperature [meters].
            z (array-like): z-coordinate(s) of the location(s) for desired temperature [meters].
            t (scalar or array-like): Time(s) at which the desired temperature is computed [seconds].

        Returns:
            numpy.ndarray: Temperature increase at the specified location(s) and time(s) in degrees Celsius.

        Example:
            .. code-block:: python

                import grheat
                import numpy as np
                import matplotlib.pyplot as plt

                # Define time, source, and observation points
                t = np.linspace(0, 500, 100) / 1000   # seconds
                x, y, z = 0, 0, 0                     # meters
                xp, yp, zp = 0, 0, 0.001              # meters
                t_pulse = 0.100                       # seconds

                # Create a Point object representing the heat source
                medium = grheat.Point(xp, yp, zp)

                # Calculate the temperature rise due to a pulsed heat source
                T = medium.pulsed(x, y, z, t, t_pulse)

                # Plot the temperature rise over time
                plt.plot(t * 1000, T, color='blue')
                plt.xlabel("Time (ms)")
                plt.ylabel("Temperature Increase (°C)")
                plt.title("1J pulse lasting %.0f ms" % t_pulse)
                plt.show()
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

        Args:
            x (array-like): x-coordinate(s) of the location(s) for desired temperature [meters].
            y (array-like): y-coordinate(s) of the location(s) for desired temperature [meters].
            z (array-like): z-coordinate(s) of the location(s) for desired temperature [meters].
            t (scalar): Time at which the desired temperature is computed [seconds].
            t_pulse (scalar): Duration of the pulse during which heat is deposited [seconds].

        Returns:
            Temperature Increase [°C]
        """
        T = self._continuous(x, y, z, t)
        if t > t_pulse:
            T -= self._continuous(x, y, z, t - t_pulse)
        return T / t_pulse

    def pulsed(self, x, y, z, t, t_pulse):
        """
        Calculates the temperature rise due to a 1J pulsed point source at specified time(s).

        This method computes the temperature rise at given location(s) `(x, y, z)` at time(s) `t`
        due to a 1J pulsed point source located at `(xp, yp, zp)`. The point source deposits heat
        from time `t=0` to `t=t_pulse`.

        Args:
            x (array-like): x-coordinate(s) of the location(s) for desired temperature [meters].
            y (array-like): y-coordinate(s) of the location(s) for desired temperature [meters].
            z (array-like): z-coordinate(s) of the location(s) for desired temperature [meters].
            t (scalar or array-like): Time(s) at which the desired temperature is computed [seconds].
            t_pulse (scalar): Duration of the pulse during which heat is deposited [seconds].

        Returns:
            numpy.ndarray: Temperature increase at the specified location(s) and time(s) in degrees Celsius.

        Example:
            .. code-block:: python

                import grheat
                import numpy as np
                import matplotlib.pyplot as plt

                # Define time, source, and observation points
                t = np.linspace(0, 500, 100) / 1000   # seconds
                x, y, z = 0, 0, 0                     # meters
                xp, yp, zp = 0, 0, 0.001              # meters
                t_pulse = 0.100                       # seconds

                # Create a Point object representing the heat source
                medium = grheat.Point(xp, yp, zp)

                # Calculate the temperature rise due to a pulsed heat source
                T = medium.pulsed(x, y, z, t, t_pulse)

                # Plot the temperature rise over time
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
