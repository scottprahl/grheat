.. |pypi-badge| image:: https://img.shields.io/pypi/v/grheat?color=68CA66
   :target: https://pypi.org/project/grheat/
   :alt: pypi

.. |github-badge| image:: https://img.shields.io/github/v/tag/scottprahl/grheat?label=github&color=68CA66
   :target: https://github.com/scottprahl/grheat
   :alt: github

.. |conda-badge| image:: https://img.shields.io/conda/vn/conda-forge/grheat?label=conda&color=68CA66
   :target: https://github.com/conda-forge/grheat-feedstock
   :alt: conda

.. |zenodo-badge| image:: https://zenodo.org/badge/533509810.svg
   :target: https://zenodo.org/badge/latestdoi/533509810
   :alt: zenodo

.. |license| image:: https://img.shields.io/github/license/scottprahl/grheat?color=68CA66
   :target: https://github.com/scottprahl/grheat/blob/main/LICENSE.txt
   :alt: License

.. |test-badge| image:: https://github.com/scottprahl/grheat/actions/workflows/test.yaml/badge.svg
   :target: https://github.com/scottprahl/grheat/actions/workflows/test.yaml
   :alt: testing

.. |docs-badge| image:: https://readthedocs.org/projects/grheat/badge?color=68CA66
   :target: https://grheat.readthedocs.io
   :alt: docs

.. |downloads| image:: https://img.shields.io/pypi/dm/grheat?color=68CA66
   :target: https://pypi.org/project/grheat/
   :alt: Downloads

.. |lite-badge| image:: https://img.shields.io/badge/try-JupyterLite-68CA66.svg
   :target: https://scottprahl.github.io/grheat/
   :alt: Try JupyterLite

grheat
======

by Scott Prahl

|pypi-badge| |github-badge| |conda-badge| |zenodo-badge|

|license| |test-badge| |docs-badge| |downloads|

|lite-badge|

Green's Functions for the Heat Equation
---------------------------------------

``grheat`` implements analytic Green's function solutions of the heat equation for a
semi-infinite medium. It is a library of closed-form temperature-rise solutions, not a
grid-based PDE solver.

The package is built around the linear heat equation

::

    rho c (dT/dt) = k nabla^2 T + q

or, equivalently,

::

    dT/dt = alpha nabla^2 T + q / (rho c)

where ``T`` is temperature rise, ``alpha`` is thermal diffusivity, and ``rho c`` is the
volumetric heat capacity of the medium.

Because this equation is linear, Green's functions act as impulse responses: once the
response to an idealized source is known, that solution can be scaled, superposed, and
integrated to model more realistic heating histories. ``grheat`` packages those analytic
solutions in a Python API for the most common source geometries used in laser-heating and
heat-transfer calculations.

In practical terms, ``grheat`` gives you direct formulas for the temperature rise caused
by:

- point sources in a half-space
- line sources parallel to the surface
- planar sources at depth
- exponentially absorbed volumetric heating from uniform surface illumination
- exponentially absorbed heating from point illumination on the surface

This makes it useful when you want fast, exact reference solutions for parameter studies,
teaching, verification of numerical models, or quick exploratory calculations.

What The Package Implements
---------------------------

The core classes correspond to canonical Green's-function source geometries:

- ``Point``: a point heat source at ``(xp, yp, zp)`` inside the medium
- ``Line``: an infinite x-directed line source passing through ``(yp, zp)``
- ``Plane``: an infinite xy-planar source at depth ``zp``
- ``ExponentialVolumeSource``: exponentially decaying volumetric heating caused by
  uniform surface illumination
- ``ExponentialColumnSource``: exponentially decaying volumetric heating caused by point
  illumination on the surface

Each class provides methods for common temporal source profiles:

- ``instantaneous(...)`` for impulsive deposition
- ``continuous(...)`` for sources that turn on and remain on
- ``pulsed(...)`` for finite-duration deposition

All methods return temperature rise, not absolute temperature.

Semi-Infinite Geometry and Boundaries
-------------------------------------

The medium occupies the half-space below the surface ``z = 0``. Surface boundary
conditions are handled analytically with the method of images. The supported boundary
conditions are:

- ``infinite``: no surface constraint
- ``adiabatic``: no heat flow across the surface, ``dT/dz = 0``
- ``zero``: the surface is held at ``T = 0``

These boundary conditions are available on the source classes where they apply.

Model Assumptions
-----------------

The analytic solutions in ``grheat`` assume:

- a homogeneous, isotropic medium
- constant thermal properties
- a semi-infinite geometry
- idealized source geometries with exact Green's-function solutions

If you need spatially varying properties, finite geometries, or arbitrary boundaries,
you will generally want a numerical PDE solver instead.

Quick Start
-----------

The example below computes the surface temperature rise caused by uniform illumination
that creates an exponentially decaying volume source. The
``ExponentialVolumeSource`` class is an analytic Green's-function solution for
volumetric heating proportional to ``mu_a exp(-mu_a z)``.

.. code-block:: python

    import grheat
    import numpy as np

    mu_a = 0.25 * 1000            # absorption coefficient [1/m]
    z = 0                         # depth [m]
    t = np.linspace(0, 0.5, 100)  # time [s]

    medium = grheat.ExponentialVolumeSource(mu_a, boundary="adiabatic")

    # Temperature rise for unit irradiance; scale by your actual irradiance as needed.
    dT = medium.continuous(z, t)

A point-source example looks like this:

.. code-block:: python

    import grheat
    import numpy as np

    source = grheat.Point(0, 0, 1e-3, boundary="zero")
    t = np.linspace(1e-3, 0.1, 200)

    # Temperature rise from a unit instantaneous point source
    dT = source.instantaneous(0, 0, 0, t)

The class and method docstrings describe the exact normalization for each source type.

Installation
------------

Install with ``pip``:

.. code-block:: bash

    pip install grheat

Or with ``conda``:

.. code-block:: bash

    conda install -c conda-forge grheat

Documentation and Examples
--------------------------

The full documentation, API reference, and example notebooks are available at
``grheat.readthedocs.io``:

    |docs-badge|

The repository also includes notebooks covering the main source classes:

- point-source examples
- line-source examples
- plane-source examples
- absorbing-medium examples

If you want to run the notebooks in the browser with no local installation, use the
JupyterLite deployment on GitHub Pages:

    |lite-badge|

If you want to open the repository directly in Google Colab, use the badge below:

.. image:: https://colab.research.google.com/assets/colab-badge.svg
  :target: https://colab.research.google.com/github/scottprahl/grheat/blob/main
  :alt: Colab

Why Green's Functions?
----------------------

Green's functions are especially useful for heat-transfer problems because they make the
structure of the solution explicit:

- the source geometry is encoded analytically
- time dependence can often be handled by convolution or finite pulse differences
- boundary conditions can be enforced exactly with image sources
- results are fast to evaluate and easy to compare against numerical simulations

That is the role of ``grheat``: it provides reusable implementations of these analytic
solutions so you can work directly with the physics of the heat equation instead of
re-deriving the formulas each time.

References
----------

The implementations are based on standard heat-conduction results, especially:

- Carslaw, H. S., and Jaeger, J. C., *Conduction of Heat in Solids*
- Prahl, Scott A., "Charts to rapidly estimate temperature following laser irradiation,"
  Proc. SPIE 2391 (1995)


License
-------

``grheat`` is licensed under the terms of the MIT license.
