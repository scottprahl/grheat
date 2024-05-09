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

grheat
======

by Scott Prahl

|pypi-badge| |github-badge| |conda-badge| |zenodo-badge|

|license| |test-badge| |docs-badge| |downloads|

Green's Functions for Heat Transfer
-----------------------------------

``grheat`` is a python module based on Green's function method for heat transfer
problems in a semi-infinite medium. There are
solutions for point sources, line sources, plane sources, and exponential heating.
Finally, the method of images is used to constrain boundary conditions at the surface
(`z=0`): 

-`infinite` (unconstrained), 
-`adiabatic` (no heat flow, dT/dz=0), and 
- `zero` (T=0).

Installation
------------

Use ``pip``::

    pip install grheat

or ``conda``::

    conda install -c conda-forge grheat

or use a Jupyter notebook immediately by clicking the Google Colaboratory button below

.. image:: https://colab.research.google.com/assets/colab-badge.svg
  :target: https://colab.research.google.com/github/scottprahl/grheat/blob/main
  :alt: Colab


License
-------

``grheat`` is licensed under the terms of the MIT license.
