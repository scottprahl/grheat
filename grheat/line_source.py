# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=consider-using-f-string
# pylint: disable=no-member
"""
Green's function heat transfer solutions for x-line source in semi-infinite media.

The surface is defined by z=0 and the line source extends horizontally
from -∞ < z < +∞.

More documentation at <https://grheat.readthedocs.io>

"""
import scipy.special
import numpy as np

water_heat_capacity = 4.184 * 1e6           # J/degree / m**3
water_thermal_diffusivity = 0.14558 * 1e-6  # m**2/s


class Line:
    """Provides Green's function heat transfer solutions for a line source.

    The line source is in a semi-infinite medium with a surface at z=0, and extends
    horizontally along all x-values passing through coordinates (yp, zp). At time tp,
    the line delivers a heat impulse of 1 Joule per meter along the line.

    Boundary conditions at z=0 can be:
        - 'infinite': No boundary.
        - 'adiabatic': No heat flow across the boundary.
        - 'zero': Boundary is fixed at T=0.
    An error is raised for any other boundary condition.
    """

    def __init__(self,
                 yp, zp,
                 tp=0,
                 diffusivity=water_thermal_diffusivity,
                 capacity=water_heat_capacity,
                 boundary='infinite'):
        """Initializes a Line object representing a line source in a medium.

        The line source extends infinitely parallel to the x-axis and passes through
        the coordinates (yp, zp) in the medium. At time tp, the line source delivers a
        heat impulse of 1 Joule per meter along the line.

        The surface of the medium is defined by z=0 and the boundary conditions may be:
            - 'infinite': No boundary.
            - 'adiabatic': No heat flow across the boundary.
            - 'zero': Boundary temperature is fixed at T=0.

        Args:
            yp (float): The y-coordinate through which the x-line source passes. [meters]
            zp (float): The z-coordinate through which the x-line source passes,
                        defining its depth below the surface z=0. [meters]
            tp (float): The time at which the line source impulse occurs. [seconds]
            diffusivity (float): The thermal diffusivity of the medium.
                                 Defaults to water_thermal_diffusivity. [m^2/s]
            capacity (float): The volumetric heat capacity of the medium.
                              Defaults to water_heat_capacity. [J/degree/m^3]
            boundary (str): string describing boundary conditions at z=0

        Raises:
            ValueError: If the specified boundary condition is not one of 'infinite',
                        'adiabatic', or 'zero'.

        Returns:
            None. Initializes the Line object.
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
            y (float): The y-coordinate for the desired temperature location. [meters]
            z (float): The z-coordinate for the desired temperature location. [meters]
            t (float): The time of the desired temperature. Should be a scalar. [seconds]
            tp (float): The time at which the line source impulse occurs.
                        Should be a scalar. [seconds]
        Returns:
            float: Normalized temperature rise.
        """
        r2 = (y - self.yp)**2 + (z - self.zp)**2

        if t <= tp:
            if np.isscalar(r2):
                return 0
            else:
                return np.zeros_like(r2)

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
            y (float): The y-coordinate for the desired temperature location. [meters]
            z (float): The z-coordinate for the desired temperature location. [meters]
            t (float): The time of the desired temperature. [seconds]

        Returns:
            float: Temperature increase in degrees Celsius [°C].

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
            if np.isscalar(self.tp):
                T = self._instantaneous(y, z, t, self.tp)
            else:
                T = np.empty_like(self.tp)
                for i, tt in enumerate(self.tp):
                    T[i] = self._instantaneous(y, z, t, tt)
        else:
            if np.isscalar(self.tp):
                T = np.empty_like(t)
                for i, tt in enumerate(t):
                    T[i] = self._instantaneous(y, z, tt, self.tp)
            else:
                raise ValueError('One of t or self.tp must be a scalar.')
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
        Calculates temperature rise due to a 1W/m continuous x-line source.

        The x-line source turns on at t=0 and passes through the coordinates (yp, zp).

        Reference: Carslaw and Jaeger (1959), page 261, Equation 10.4(5).

        Parameters:
            y (float): The y-coordinate for the desired temperature location. [meters]
            z (float): The z-coordinate for the desired temperature location. [meters]
            t (float or array-like): Time(s) of desired temperature. [seconds]

        Returns:
            float or numpy.ndarray: Temperature increase in degrees Celsius [°C].

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
            T = self._continuous(y, z, t)

        else:
            T = np.empty_like(t)
            for i, tt in enumerate(t):
                T[i] = self._continuous(y, z, tt)
        return T

    def _pulsed(self, y, z, t, t_pulse):
        """
        Calculates temperature rise due to a 1 J/m pulsed x-line source at time(s) t.

        This method computes the temperature rise at a specified location and time due
        to a pulsed x-line source. 1 J/m of heat is deposited along the x-line passing
        through the coordinates (yp, zp) from t=0 to t=t_pulse.

        Parameters:
            y (float): The y-coordinate for the desired temperature location. [meters]
            z (float): The z-coordinate for the desired temperature location. [meters]
            t (float): Time of desired temperature. [seconds]
            t_pulse (float): Duration of the pulse. [seconds]

        Returns:
            float: Temperature increase in degrees Celsius [°C].

        Raises:
            ValueError: If `t_pulse` is negative.
        """
        if t_pulse < 0:
            raise ValueError("Pulse duration (%f) must be positive" % t_pulse)

        T = self._continuous(y, z, t)
        if t > t_pulse:
            T -= self._continuous(y, z, t - t_pulse)
        return T / t_pulse

    def pulsed(self, y, z, t, t_pulse):
        """
        Calculates temperature rise due to a 1 J/m pulsed x-line source.

        This method computes the temperature rise at a specified location and time due
        to a pulsed x-line source. 1 J/m of heat is deposited along the x-line passing
        through the coordinates (yp, zp) from t=0 to t=t_pulse.

        Parameters:
            y (float): The y-coordinate for the desired temperature location. [meters]
            z (float): The z-coordinate for the desired temperature location. [meters]
            t (float or array-like): Time(s) of desired temperature. [seconds]
            t_pulse (float): Duration of the pulse. [seconds]

        Returns:
            float or numpy.ndarray: Temperature increase in degrees Celsius [°C].

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
        if np.isscalar(t):
            T = self._pulsed(y, z, t, t_pulse)
        else:
            T = np.empty_like(t)
            for i, tt in enumerate(t):
                T[i] = self._pulsed(y, z, tt, t_pulse)
        return T
