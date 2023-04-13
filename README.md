
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

0.1.2
-------

Corrected segfault resulting from initialization from generators that raise.


0.1.1
-------

Added `__version__` to module; releasing wheels.


0.1.0
-------

Initial release with NumPy integration.

