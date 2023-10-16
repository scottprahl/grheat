# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=consider-using-f-string
# pylint: disable=no-member
"""
Green's Function Heat Transfer Solutions for xy-Planar Source in Infinite Media
===============================================================================

This module provides Green's function solutions for heat transfer due to an xy-planar
source in an infinite medium, encapsulated within the `Plane` class. The solutions are
based on the mathematical formulations provided in Carslaw and Jaeger's work.

The `Plane` class represents a planar heat source located at a specified depth `zp` in
the medium. It provides methods to calculate the temperature rise at any given depth `z`
at a specified time `t` due to different types of heat source behavior.

Three types of planar sources are supported:

- **Instantaneous**: Represents a single, instantaneous release of heat from zp-plane at time `tp`.

- **Continuous**: Represents a continuous release of heat from the zp-plane starting at t=0

- **Pulsed**: Represents a pulsed release of heat from the zp-plane for t=0 to `t_pulse`.

The module supports various boundary conditions such as infinite, adiabatic, or zero
boundary, and allows for specifying thermal properties like diffusivity and volumetric
heat capacity of the medium.

More documentation can be found at `grheat Documentation <https://grheat.readthedocs.io>`_.

"""

import scipy.special
import numpy as np

water_heat_capacity = 4.184 * 1e6           # J/degree / m**3
water_thermal_diffusivity = 0.14558 * 1e-6  # m**2/s


class Plane:
    """
    Green's Function Heat Transfer Solutions for xy-Planar Source in Infinite Media.

    This class encapsulates the Green's function solutions for an xy-planar heat source
    in an infinite medium. It provides methods for calculating the temperature rise at a
    specified depth `z` and time `t` due to different behaviors of the heat source:
    instantaneous, continuous, or pulsed.

    The `Plane` object should be initialized with the depth of the xy-planar source `zp`,
    and optionally, the time of source impulse `tp`, thermal diffusivity, volumetric heat
    capacity, and boundary condition can also be specified.

    Attributes:
        zp (scalar or array): Depth of the xy-planar source [meters].
        tp (scalar or array): Time of source impulse [seconds]. Default is 0.
        diffusivity (float): Thermal diffusivity [m**2/s].
        capacity (float): Volumetric heat capacity [J/degree/m**3].
        boundary (str): Boundary condition, one of 'infinite', 'adiabatic', or 'zero'.
    """
    def __init__(self,
                 zp, tp=0,
                 diffusivity=water_thermal_diffusivity,
                 capacity=water_heat_capacity,
                 boundary='infinite'):
        """
        Initializes a Plane object for a planar heat source in an semi-infinite medium.

        This method initializes the Plane object with the specified properties and settings
        for the planar heat source, including its depth, time of occurrence, thermal
        properties of the medium, and boundary condition. These settings will be used
        in subsequent calculations of temperature rise due to the planar source.

        Args:
            zp (float): Depth of the xy-planar source in meters.
            tp (float, optional): Time of the source impulse in seconds. Defaults to 0.
            diffusivity (float, optional): Thermal diffusivity of the medium in m**2/s.
                Defaults to `water_thermal_diffusivity`.
            capacity (float, optional): Volumetric heat capacity of the medium in J/degree/m**3.
                Defaults to `water_heat_capacity`.
            boundary (str, optional): Boundary condition of the medium.
                Options include 'infinite', 'adiabatic', or 'zero'. Defaults to 'infinite'.

        Raises:
            ValueError: If an invalid value is provided for the `boundary` argument.
        """
        self.zp = zp

        self.tp = tp
        self.diffusivity = diffusivity     # m**2/s
        self.capacity = capacity           # J/degree/kg
        self.boundary = boundary.lower()
        if self.boundary not in ['infinite', 'adiabatic', 'zero']:
            raise ValueError("boundary must be 'infinite', 'adiabatic', or 'zero'")

    def _instantaneous(self, z, t, tp):
        """
        Computes the temperature rise due to a 1 J/m² instantaneous xy-planar source at time `t`.

        This method calculates the temperature increase at a specified depth `z` at time `t`
        due to an instantaneous planar heat source of 1 J/m² occurring at time `tp`.
        The source plane is parallel to the surface and passes through depth `zp`,
        as specified during the initialization of the `Plane` object.

        The formulation is based on Carslaw and Jaeger (page 259, equation 10.3(4)).

        Args:
            z (float or array-like): Depth(s) of desired temperature in meters.
            t (float): Time of desired temperature in seconds.
            tp (float): Time of source impulse in seconds.

        Returns:
            float or numpy.ndarray: Temperature increase in degrees Celsius at the specified depth(s) and time.

        The method handles scalar and array-like input for the depth `z`.
        When `z` is a scalar, a single float representing the temperature increase is returned.
        When `z` is array-like, a NumPy array of temperature increases corresponding to each depth value is returned.
        """
        z2 = (z - self.zp)**2
        if t <= tp:
            if np.isscalar(z2):
                return 0
            else:
                return np.zeros_like(z2)

        factor = self.capacity * 2 * np.sqrt(np.pi * self.diffusivity * (t - tp))
        T = 1 / factor * np.exp(-z2 / (4 * self.diffusivity * (t - tp)))

        if self.boundary != 'infinite':
            z2 = (z + self.zp)**2
            T1 = 1 / factor * np.exp(-z2 / (4 * self.diffusivity * (t - tp)))

            if self.boundary == 'adiabatic':
                T += T1

            if self.boundary == 'zero':
                T -= T1

        return T

    def instantaneous(self, z, t):
        """
        Computes the temperature rise due to a 1 J/m² instant xy-planar source.

        This method calculates the temperature increase at a specified depth `z` at time(s) `t`
        due to an instantaneous planar heat source of 1 J/m². The source plane is parallel
        to the surface and passes through depth `zp`, as specified during the initialization
        of the `Plane` object. The formulation is based on Carslaw and Jaeger
        (page 259, equation 10.3(4)).

        This method handles scalar and array-like input for time `t`. When `t` is a scalar,
        a single float representing the temperature increase is returned. When `t` is array-like,
        a NumPy array of temperature increases corresponding to each time value is returned.

        Args:
            z (float or array-like): Depth(s) for desired temperature in meters.
            t (scalar or array-like): Time(s) of desired temperature in seconds.

        Raises:
            ValueError: If both `t` and `tp` are arrays. One of them must be a scalar.

        Returns:
            float or array: Temp increase in degrees Celsius at the depth(s) and time(s).

        Example:

        .. code-block:: python

            import grheat
            import numpy as np
            import matplotlib.pyplot as plt

            t = np.linspace(0, 500, 100) / 1000   # seconds
            z = 0                                 # meters
            zp = 0.001                            # meters

            plane = grheat.Plane(zp)
            T = plane.instantaneous(z, t)

            plt.plot(t * 1000, T, color='blue')
            plt.xlabel("Time (ms)")
            plt.ylabel("Temperature Increase (°C)")
            plt.title("Temperature Rise due to 1 J/m² Instant Planar Source")
            plt.show()

        """
        if np.isscalar(t):
            if np.isscalar(self.tp):
                T = self._instantaneous(z, t, self.tp)
            else:
                T = np.empty_like(self.tp)
                for i, tt in enumerate(self.tp):
                    T[i] = self._instantaneous(z, t, tt)
        else:
            if np.isscalar(self.tp):
                T = np.empty_like(t)
                for i, tt in enumerate(t):
                    T[i] = self._instantaneous(z, tt, self.tp)
            else:
                raise ValueError('One of t or self.tp must be a scalar.')
        return T

    def _continuous(self, z, t):
        """
        Computes the temperature rise due to a continuous 1 W/m² xy-planar heat source.

        The heat source, positioned at a depth of `zp`, initiates at t=0 and continues
        until the specified time `t`. The computation is achieved by integrating the
        planar Green's function from 0 to `t` using Mathematica.

        Args:
            z (float or array-like): Depth(s) at which the temperature is desired [meters].
            t (float): Time(s) at which the temperature is evaluated [seconds].

        Returns:
            float or numpy.ndarray: The temperature increase(s) at the specified depth(s)
            and time(s) in degrees Celsius. The return type matches the input type (scalar
            for scalar input, array for array input).
        """
        dz = z - self.zp
        if t <= 0:
            if np.isscalar(dz):
                return 0
            else:
                return np.zeros_like(dz)

        alpha = np.sqrt(dz**2 / (4 * self.diffusivity * t))
        T = np.exp(-alpha**2) / np.sqrt(np.pi) - alpha * scipy.special.erfc(alpha)
        T *= np.sqrt(t / self.diffusivity) / self.capacity

        if self.boundary != 'infinite':
            alpha = np.sqrt((z + self.zp)**2 / (4 * self.diffusivity * t))
            T1 = np.exp(-alpha**2) / np.sqrt(np.pi) - alpha * scipy.special.erfc(alpha)
            T1 *= np.sqrt(t / self.diffusivity) / self.capacity

            if self.boundary == 'adiabatic':
                T += T1

            if self.boundary == 'zero':
                T -= T1

        return T

    def continuous(self, z, t):
        """
        Computes the temperature rise due to a continuous 1W/m² xy-planar heat source.

        The heat source, situated at a depth of `zp`, initiates at t=0 and persists
        continuously until the specified time `t`. This method serves as a wrapper
        for the `_continuous` method, facilitating handling of scalar or array-like
        input for time `t`.

        Args:
            z (float or array-like): Depth(s) at which the temperature is desired [meters].
            t (float or array-like): Time(s) at which the temperature is evaluated [seconds].

        Returns:
            float or numpy.ndarray: The temperature increase(s) at the specified depth(s)
            and time(s) in degrees Celsius. The return type matches the input type (scalar
            for scalar input, array for array input).

        Example:

            .. code-block:: python

                import grheat
                import numpy as np
                import matplotlib.pyplot as plt

                t = np.linspace(0, 500, 100) / 1000   # seconds
                z = 0                                 # meters
                zp = 0.001                            # meters

                plane = grheat.Plane(zp=zp)
                T = plane.continuous(z, t)

                plt.plot(t * 1000, T, color='blue')
                plt.xlabel("Time (ms)")
                plt.ylabel("Temperature Increase (°C) at z=0")
                plt.title("Continuous 1W/m² source 1mm deep")
                plt.show()
        """
        if np.isscalar(t):
            T = self._continuous(z, t)

        else:
            T = np.empty_like(t)
            for i, tt in enumerate(t):
                T[i] = self._continuous(z, tt)
        return T

    def _pulsed(self, z, t, t_pulse):
        """
        Calculate temperature rise due to a 1 J/m² pulsed xy-planar source at time t.

        1 J/m² of heat deposited in an xy-plane passing at zp from t=0 to t=t_pulse.

        Args:
            z: depth for desired temperature [meters]
            t: time of desired temperature [seconds]
            t_pulse: duration of pulse [seconds]

        Returns:
            Temperature increase [°C]
        """
        if t_pulse < 0:
            raise ValueError("Pulse duration (%f) must be positive" % t_pulse)

        T = self._continuous(z, t)
        if t > t_pulse:
            T -= self._continuous(z, t - t_pulse)
        return T / t_pulse

    def pulsed(self, z, t, t_pulse):
        """
        Computes the temperature rise due to a 1 J/m² pulsed xy-planar heat source.

        The xy-planar source, situated at a depth of `zp`, emits a pulse of 1 J/m²
        from time t=0 to time t=`t_pulse`. This method acts as a wrapper for the
        `_pulsed` method, accommodating scalar or array-like input for time `t`.

        Args:
            z (float or array-like): Depth(s) at which the temperature is desired [meters].
            t (float or array-like): Time(s) at which the temperature is evaluated [seconds].
            t_pulse (float): Duration of the heat pulse [seconds].

        Returns:
            float or numpy.ndarray: The temperature increase(s) at the specified depth(s)
            and time(s) in degrees Celsius. The return type matches the input type (scalar
            for scalar input, array for array input).

        Raises:
            ValueError: If `t_pulse` is not a positive number.

        Example:

            .. code-block:: python

                import grheat
                import numpy as np
                import matplotlib.pyplot as plt

                t = np.linspace(0, 500, 100) / 1000   # seconds
                z = 0                                 # meters
                zp = 0.001                            # meters
                t_pulse = 0.100                       # seconds

                plane = grheat.Plane(zp=zp)
                T = plane.pulsed(z, t, t_pulse)

                plt.plot(t * 1000, T, color='blue')
                plt.xlabel("Time (ms)")
                plt.ylabel("Temperature Increase at z=0 (°C)")
                plt.title("1J pulse lasting %.0f ms" % t_pulse)
                plt.show()
        """
        if np.isscalar(t):
            T = self._pulsed(z, t, t_pulse)
        else:
            T = np.empty_like(t)
            for i, tt in enumerate(t):
                T[i] = self._pulsed(z, tt, t_pulse)
        return T
