"""
Green's Functions for Heat Transfer in grheat package.

`grheat` is a Python module that utilizes the Green's function method to solve heat transfer
problems within a semi-infinite medium. This method provides solutions for different types of
heat sources including point sources, line sources, plane sources, and exponential heating.
Additionally, the method of images is employed to handle boundary conditions at the surface (`z=0`)
with the following constraints:

- `infinite` (unconstrained),
- `adiabatic` (no heat flow, dT/dz=0), and
- `zero` (T=0).

Classes
-------
- `ExponentialVolumeSource` : Uniform surface illumination producing an
  exponentially depth-weighted volume source.
- `Point` : A point source of heat in the medium.
- `Line` : A line source of heat, parallel to the surface, in the medium.
- `Plane` : A plane source of heat in the medium.
- `ExponentialColumnSource`: Exponentially depth-weighted heating from point
  illumination on the surface.

Complete documentation, including detailed descriptions of classes and methods, is available at
<https://grheat.readthedocs.io>

Example Usage
-------------
.. code-block:: python

    import grheat
    import numpy as np

    mu_a = 0.25 * 1000            # 1/m  (absorption coefficient)
    z = 0                         # meters
    t = np.linspace(0, 0.5, 100)  # seconds

    # Create an ExponentialVolumeSource instance
    medium = grheat.ExponentialVolumeSource(mu_a)

    # Compute temperature rise due to continuous irradiance
    T = medium.continuous(z, t)
"""

__version__ = "0.2.1"
__author__ = "Scott Prahl"
__email__ = "scott.prahl@oit.edu"
__copyright__ = "2022-23, Scott Prahl"
__license__ = "MIT"
__url__ = "https://github.com/scottprahl/grheat"

from .point_source import *
from .line_source import *
from .plane_source import *
from .exponential_volume_source import *
from .exponential_column_source import *
