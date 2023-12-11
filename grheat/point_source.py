# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=consider-using-f-string
# pylint: disable=no-member
"""
Heat transfer Green's functions for a point source in a semi-infinite medium.

The `Point` class represents a point heat source located at a specified position
`(xp, yp, zp)` in the medium. It provides methods to calculate the temperature
rise at any given location `(x, y, z)` at a specified time `t` due to different
types of heat source behavior.

Three types of point sources are supported:

- **Instantaneous**: Instantaneous release of heat from `(xp,yp,zp)` at time(s) `tp`.

- **Continuous**: Continuous release of heat from `(xp,yp,zp)` at time(s) `tp`.

- **Pulsed**: Pulsed release of heat from `(xp,yp,zp)` with pulse start at time(s) `tp`.

Each of these illumination types can be analyzed for boundary conditions at ``z=0``:

- `'infinite'`: No boundary (infinite medium).
- `'adiabatic'`: No heat flow across the boundary.
- `'zero'`: Boundary is fixed at ``T=0``.

The module allows for specifying thermal properties like diffusivity and
volumetric heat capacity of the medium.

The solutions are based on the mathematical formulations provided in Carslaw and
Jaeger's work.

More documentation can be found at <https://grheat.readthedocs.io>
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
                 xp, yp, zp, tp=0,
                 diffusivity=water_thermal_diffusivity,
                 capacity=water_heat_capacity,
                 boundary='infinite'):
        """
        Initialize a Point object representing a point heat source in a medium.

        This method sets up a point heat source at a specified location, with specified
        thermal properties and boundary conditions.

        Args:
            xp (scalar): The x-coordinate of the point source in meters.
            yp (scalar): The y-coordinate of the point source in meters.
            zp (scalar): The z-coordinate of the point source in meters.
            tp (scalar): Time at which the source impulse occurs [seconds].
            diffusivity (scalar, optional): The thermal diffusivity of the medium in
                square meters per second (m**2/s). Defaults to water_thermal_diffusivity.
            capacity (scalar, optional): The volumetric heat capacity of the medium in
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
        self.tp = tp
        self.diffusivity = diffusivity
        self.capacity = capacity
        self.boundary = boundary.lower()
        if self.boundary not in ['infinite', 'adiabatic', 'zero']:
            raise ValueError("boundary must be 'infinite', 'adiabatic', or 'zero'")

    def __str__(self):
        """Create string for object."""
        return (f"Point Properties:\n"
                f"xp: {self.xp} meters\n"
                f"yp: {self.yp} meters\n"
                f"zp: {self.zp} meters\n"
                f"tp: {self.tp} seconds\n"
                f"diffusivity: {self.diffusivity} m^2/s\n"
                f"capacity: {self.capacity} J/degree/m^3\n"
                f"boundary: {self.boundary}\n")

    def _instantaneous(self, x, y, z, t, tp):
        """
        Calculate temperature rise due to a 1J instant point source at a specific time `t`.

        This method computes the temperature rise at a given location `(x, y, z)` at time `t`
        due to a 1J instantaneous point source occurring at time `tp`, following the formula
        from Carslaw and Jaeger (page 256, 10.2(2)). Both `t` and `tp` are assumed to be scalars.

        Args:
            x (array): x-coordinate(s) of the location(s) for desired temperature [meters].
            y (array): y-coordinate(s) of the location(s) for desired temperature [meters].
            z (array): z-coordinate(s) of the location(s) for desired temperature [meters].
            t (scalar): Time at which the desired temperature is computed [seconds].

        Returns:
            array: Normalized temperature rise at the specified location(s) and time.
        """
        r = np.sqrt((x - self.xp)**2 + (y - self.yp)**2 + (z - self.zp)**2)
        if t <= tp:
            return r * 0

        dt = self.diffusivity * (t - tp)
        factor = self.capacity * 8 * (np.pi * dt)**1.5
        T = 1 / factor * np.exp(-r**2 / (4 * dt))

        if self.boundary != 'infinite':
            r = np.sqrt((x - self.xp)**2 + (y - self.yp)**2 + (z + self.zp)**2)
            factor = self.capacity * 8 * (np.pi * dt)**1.5
            T1 = 1 / factor * np.exp(-r**2 / (4 * dt))

            if self.boundary == 'adiabatic':
                T += T1

            if self.boundary == 'zero':
                T -= T1

        return T

    def instantaneous(self, x, y, z, t):
        """
        Calculate the temperature rise due to a 1J instant point source at specified time(s).

        This method computes the temperature rise at given location(s) `(x, y, z)` at time(s) `t`
        due to a 1J instantaneous point source located at `(xp, yp, zp)` occurring at time(s) `tp`,
        following the formula from Carslaw and Jaeger (page 256, 10.2(2)). Either `t` or `tp`
        should be a scalar while the other can be an array.

        Args:
            x (scalar or array): x-coord(s) of desired temperature [meters].
            y (scalar or array): y-coord(s) of desired temperature [meters].
            z (scalar or array): z-coord(s) of desired temperature [meters].
            t (scalar or array): Time(s) at which the desired temperature is computed [seconds].

        Returns:
            scalar or array: Temperature increase in °C at the specified location(s) and time(s).

        Example:

        .. code-block:: python

            import grheat
            import numpy as np
            import matplotlib.pyplot as plt

            # Define time, source, and observation points
            t = np.linspace(10, 500, 100) / 1000  # seconds
            x, y, z = 0, 0, 0                     # meters
            xp, yp, zp = 0, 0, 0.001              # meters
            t_p = 0                               # seconds

            # Create a Point object representing the heat source
            point = grheat.Point(xp, yp, zp, tp)

            # Calculate the temperature rise due to a pulsed heat source
            T = point.instantaneous(x, y, z, t)

            # Plot the temperature rise over time
            plt.plot(t * 1000, T, color='blue')
            plt.xlabel("Time (ms)")
            plt.ylabel("Temperature Increase (°C)")
            plt.title("1J pulse lasting %.0f ms" % t_pulse)
            plt.show()
        """
        if np.isscalar(t):
            T = 0.0                # return a scalar
            if np.isscalar(self.tp):
                T += self._instantaneous(x, y, z, t, self.tp)
            else:
                for i, tp in enumerate(self.tp):
                    T += self._instantaneous(x, y, z, t, tp)

        else:
            T = np.zeros_like(t, dtype=float)  # return an array
            for i, tt in enumerate(t):
                if np.isscalar(self.tp):
                    T[i] += self._instantaneous(x, y, z, tt, self.tp)
                else:
                    for tp in self.tp:
                        T[i] += self._instantaneous(x, y, z, tt, tp)
        return T

    def distance(self, x, y, z):
        """Calculate distance to the source --- avoid returning zero."""
        r = np.sqrt((x - self.xp)**2 + (y - self.yp)**2 + (z - self.zp)**2)
        rr = np.asarray(r)
        rr[rr == 0] = 1e-6  # 1 micron
        if rr.shape == ():
            return rr.item()
        return rr

    def _continuous(self, x, y, z, t):
        """
        Calculate temperature rise of a 1W continuous point source at time t.

        Carslaw and Jaeger page 261, 10.4(2)

        Parameters:
            x (scalar or array): x-coord(s) of desired temperature [meters].
            y (scalar or array): y-coord(s) of desired temperature [meters].
            z (scalar or array): z-coord(s) of desired temperature [meters].
            t (scalar): time of desired temperature [seconds]

        Returns:
            Temperature Increase [°C]
        """
        r = self.distance(x, y, z)
        if t <= 0:
            return r * 0

        factor = 1 / self.capacity / (4 * np.pi * self.diffusivity * r)
        T = factor * scipy.special.erfc(r / np.sqrt(4 * self.diffusivity * t))

        if self.boundary != 'infinite':
            r = self.distance(x, y, -z)
            factor = 1 / self.capacity / (4 * np.pi * self.diffusivity * r)
            T1 = factor * scipy.special.erfc(r / np.sqrt(4 * self.diffusivity * t))

            if self.boundary == 'adiabatic':
                T += T1

            if self.boundary == 'zero':
                T -= T1

        return T

    def continuous(self, x, y, z, t):
        """
        Calculate the temperature rise due to a 1W continuous point source at specified time(s).

        This method computes the temperature rise at given location(s) `(x, y, z)` at time(s) `t`
        due to a 1W continuous point source located at `(xp, yp, zp)` that was turned on at
        time `t=0`, following the formula from Carslaw and Jaeger (page 261, 10.4(2)).

        Args:
            x (scalar or array): x-coord(s) of desired temperature [meters].
            y (scalar or array): y-coord(s) of desired temperature [meters].
            z (scalar or array): z-coord(s) of desired temperature [meters].
            t (scalar or array): Time(s) for temperature to be computed [seconds].

        Returns:
            scalar or array: Temperature increase (°C) at the specified location(s) and time(s).

        Example:

            .. code-block:: python

                import grheat
                import numpy as np
                import matplotlib.pyplot as plt

                # Define time, source, and observation points
                t = np.linspace(0, 500, 100) / 1000   # seconds
                x, y, z = 0, 0, 0                     # meters
                xp, yp, zp = 0, 0, 0.001              # meters

                # Create a Point object representing the heat source
                point = grheat.Point(xp, yp, zp)

                # Calculate the temperature rise due to a pulsed heat source
                T = point.continuous(x, y, z, t)

                # Plot the temperature rise over time
                plt.plot(t * 1000, T, color='blue')
                plt.xlabel("Time (ms)")
                plt.ylabel("Temperature Increase (°C)")
                plt.title("1J pulse lasting %.0f ms" % t_pulse)
                plt.show()
        """
        if np.isscalar(t):
            T = 0.0                # return a scalar
            if np.isscalar(self.tp):
                T += self._continuous(x, y, z, t - self.tp)
            else:
                for i, tp in enumerate(self.tp):
                    T += self._continuous(x, y, z, t - tp)

        else:
            T = np.zeros_like(t, dtype=float)  # return an array
            for i, tt in enumerate(t):
                if np.isscalar(self.tp):
                    T[i] += self._continuous(x, y, z, tt - self.tp)
                else:
                    for tp in self.tp:
                        T[i] += self._continuous(x, y, z, tt - tp)
        return T

    def pulsed(self, x, y, z, t, t_pulse):
        """
        Calculate temperature rise due to a 1J pulsed point source at time(s) t.

        This method computes the temperature rise at given location(s) `(x, y, z)`
        at time(s) `t` due to a 1J pulsed point source located at `(xp, yp, zp)`.
        The point source deposits heat from time `t=self.tp` to `t=self.tp+t_pulse`.

        Args:
            x (scalar or array): x-coord(s) of desired temperature [meters].
            y (scalar or array): y-coord(s) of desired temperature [meters].
            z (scalar or array): z-coord(s) of desired temperature [meters].
            t (scalar or array): Time(s) at which the desired temperature is computed [seconds].
            t_pulse (scalar): Duration of the pulse during which heat is deposited [seconds].

        Returns:
            scalar or array: Temperature increase (°C) at the location(s) and time(s).

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
        if t_pulse < 0:
            raise ValueError("Pulse duration (%f) must be positive" % t_pulse)

        T = self.continuous(x, y, z, t)
        tt = np.subtract(t, t_pulse, dtype=np.float64)
        T -= self.continuous(x, y, z, tt)
        return T / t_pulse
