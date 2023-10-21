grheat
======

by Scott Prahl

.. image:: https://img.shields.io/pypi/v/grheat?color=68CA66
   :target: https://pypi.org/project/grheat/
   :alt: pypi

.. image:: https://img.shields.io/github/v/tag/scottprahl/grheat?label=github&color=68CA66
   :target: https://github.com/scottprahl/grheat
   :alt: github

.. image:: https://img.shields.io/conda/vn/conda-forge/grheat?label=conda&color=68CA66
   :target: https://github.com/conda-forge/grheat-feedstock
   :alt: conda

.. image:: https://zenodo.org/badge/533509810.svg
   :target: https://zenodo.org/badge/latestdoi/533509810
   :alt: zenodo

|

.. image:: https://img.shields.io/github/license/scottprahl/grheat?color=68CA66
   :target: https://github.com/scottprahl/grheat/blob/master/LICENSE.txt
   :alt: License

.. image:: https://github.com/scottprahl/grheat/actions/workflows/test.yaml/badge.svg
   :target: https://github.com/scottprahl/grheat/actions/workflows/test.yaml
   :alt: testing

.. image:: https://readthedocs.org/projects/grheat/badge?color=68CA66
  :target: https://grheat.readthedocs.io
  :alt: docs

.. image:: https://img.shields.io/pypi/dm/grheat?color=68CA66
   :target: https://pypi.org/project/grheat/
   :alt: Downloads

__________

Green's Functions for Heat Transfer
-----------------------------------

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
  recognized literature (e.g., Carslaw and Jaeger).

Installation
------------

**these don't work yet**

Use ``pip``::

    pip install grheat

or ``conda``::

    conda install -c conda-forge grheat

or use a Jupyter notebook immediately by clicking the Google Colaboratory button below

.. image:: https://colab.research.google.com/assets/colab-badge.svg
  :target: https://colab.research.google.com/github/scottprahl/grheat/blob/master
  :alt: Colab


License
-------

``grheat`` is licensed under the terms of the MIT license.
