"""
Green's Functions for Heat Transfer.

`grheat` is a library offering Green's function solutions for heat transfer
problems in various geometries and boundary conditions. This module provides
solutions for point sources, line sources, plane sources, and exponential heating
in semi-infinite media.

Main Features
-------------
- Green's function solutions for point, line, and plane sources in semi-infinite media.
- Solutions for exponential heating of a semi-infinite absorbing medium.
- Supports different boundary conditions at the surface (z=0): infinite, adiabatic, and zero.
- Accurate and efficient calculations based on mathematical formulations provided in
  recognized literature (e.g., Carslaw and Jaeger, Prahl's 1995 SPIE paper).

Documentation
-------------
Complete documentation is available at <https://grheat.readthedocs.io>

Example Usage
-------------
.. code-block:: python

    import grheat
    import numpy as np

    # Define parameters
    mu_a = 0.25 * 1000  # 1/m
    z = 0  # meters
    t = np.linspace(0, 500, 100) / 1000  # seconds

    # Create an Absorber instance
    medium = grheat.Absorber(mu_a)

    # Compute temperature rise due to continuous irradiance
    T = medium.continuous(z, t)
"""

__version__ = '0.2.0'
__author__ = 'Scott Prahl'
__email__ = 'scott.prahl@oit.edu'
__copyright__ = '2022-23, Scott Prahl'
__license__ = 'MIT'
__url__ = 'https://github.com/scottprahl/grheat'

from .point_source import *
from .line_source import *
from .plane_source import *
from .absorber import *
