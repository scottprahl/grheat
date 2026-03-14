.. |pypi-badge| image:: https://img.shields.io/pypi/v/grheat?color=68CA66
   :target: https://pypi.org/project/grheat/
   :alt: pypi

.. |github-badge| image:: https://img.shields.io/github/v/tag/scottprahl/grheat?label=github&color=68CA66
   :target: https://github.com/scottprahl/grheat
   :alt: github

.. |conda-badge| image:: https://img.shields.io/conda/vn/conda-forge/grheat?label=conda&color=68CA66
   :target: https://github.com/conda-forge/grheat-feedstock
   :alt: conda

.. |zenodo-badge| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.10030670.svg
   :target: https://doi.org/10.5281/zenodo.10030670
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

``grheat`` implements analytic Green's function solutions of the heat equation::

   ρ c ∂T/∂t = k ∇²T + q

or::

    ∂T/∂t = 𝛼 ∇² T + q/(ρ c)

for a semi-infinite medium. It is a library of closed-form temperature-rise solutions, not a
grid-based PDE solver. 


Green's functions are especially useful for heat-transfer problems because they make the
structure of the solution explicit:

- the source geometry is encoded analytically
- time dependence can often be handled by convolution or finite pulse differences
- boundary conditions can be enforced exactly with image sources
- results are fast to evaluate and easy to compare against numerical simulations

That is the role of ``grheat``: it provides reusable implementations of these analytic
solutions so you can work directly with the physics of the heat equation instead of
re-deriving the formulas each time.

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

The medium occupies the half-space below the surface ``z = 0`` and surface boundary
conditions are handled analytically with the method of images. The supported boundary
conditions are:

- ``infinite``: no surface constraint
- ``adiabatic``: no heat flow across the surface, ``dT/dz = 0``
- ``zero``: the surface is held at ``T = 0``

All methods return temperature rise, not absolute temperature.

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

Full documentation, API reference, and example notebooks are available at
``grheat.readthedocs.io``:

    |docs-badge|


Installation
------------

Install with ``pip``:

.. code-block:: bash

    pip install grheat

Or with ``conda``:

.. code-block:: bash

    conda install -c conda-forge grheat

If you want to run the notebooks in the browser with no local installation, use the
JupyterLite deployment on GitHub Pages:

    |lite-badge|


Documentation and Examples
--------------------------

The full documentation, API reference, and example notebooks are available at
``grheat.readthedocs.io``:

    |docs-badge|

If you want to run the notebooks in the browser with no local installation, use the
JupyterLite deployment on GitHub Pages:

    |lite-badge|

Citation
--------

If you use ``grheat`` in academic, instructional, or applied technical work, please cite:

Prahl, S. (2026). *grheat: a python module for heat transfer Green's functions* (Version 0.5.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.10030670

BibTeX:

.. code-block:: bibtex

    @software{prahl_grheat_2026,
      author  = {Scott Prahl},
      title   = {{grheat}: a python module for heat transfer {G}reen's functions},
      url     = {https://github.com/scottprahl/grheat},
      doi     = {10.5281/zenodo.10030670},
      year    = {2026},
      version = {0.5.0},
      publisher = {Zenodo}
    }


References
----------

The implementations are based on standard heat-conduction results, especially:

- Carslaw, H. S., and Jaeger, J. C., *Conduction of Heat in Solids*, (1959).
- Prahl, Scott A., "Charts to rapidly estimate temperature following laser irradiation,"
  Proc. SPIE 2391 (1995)


License
-------

``grheat`` is licensed under the terms of the MIT license.
