

.. image:: https://img.shields.io/pypi/pyversions/arraymap.svg
  :target: https://pypi.org/project/arraymap

.. image:: https://img.shields.io/pypi/v/arraymap.svg
  :target: https://pypi.org/project/arraymap

.. image:: https://img.shields.io/github/actions/workflow/status/static-frame/arraymap/ci.yml?branch=master&label=build&logo=Github
  :target: https://github.com/static-frame/arraymap/actions/workflows/ci.yml



arraymap
============

The ArrayMap library provides dictionary-like lookup from NumPy array values to their integer positions. The hash table design and C implementation is based on [AutoMap](https://github.com/brandtbucher/automap), with extensive additions for direct support of NumPy arrays.


Code: https://github.com/static-frame/arraymap

Packages: https://pypi.org/project/arraymap



Dependencies
--------------

ArrayMap requires the following:

- Python >= 3.7
- NumPy >= 1.18.5



What is New in ArrayMap
-------------------------

0.1.5
........

Improved handling for Unicode elements that contain non-terminal NULL strings.


0.1.4
........

Corrected comparison in lookup of Unicode elements.


0.1.3
........

Updated ``classifiers``, ``install_requires`` with ``setuptools``.


0.1.2
........

Corrected segfault resulting from initialization from generators that raise.


0.1.1
........

Added `__version__` to module; releasing wheels.


0.1.0
........

Initial release with NumPy integration.

