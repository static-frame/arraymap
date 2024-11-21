

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

- Python >= 3.9
- NumPy >= 1.19.5



What is New in ArrayMap
-------------------------

0.4.0
........

Now building wheels for Python 3.13.


0.3.0
........

Now building with NumPy 2.0.


0.2.2
........

Restored functional wheels for Mac OS ``x86_64``.


0.2.1
........

Restored functional wheels for Mac OS ``x86_64``.


0.2.0
........

Now building wheels for 3.12.

Now building functional wheels for Mac OS ``arm64`` / Apple Silicon.


0.1.9
........

Improvements to ``PyObject`` struct layout and other internal refactoring.


0.1.8
........

Corrected issue when using ``get_all()`` and ``get_any()`` on ``FrozenAutoMap`` backed by numerical arrays with less than 64-bit element size.


0.1.7
........

Corrected issue when creating a ``FrozenAutoMap`` from a ``datetime64`` array that has duplicates.


0.1.6
........

Implemented ``get_all()`` and ``get_any()`` for optimized lookup of multiple keys from arrays or lists.

Implemented full support for ``np.datetime64`` arrays.


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

