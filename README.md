
arraymap
============

`arraymap` is a Python package containing high-performance autoincremented
integer-valued mappings.

To install, just run `pip install arraymap`.

Examples
--------

`arraymap` objects are sort of like "inverse sequences". They come in two
variants:

### FrozenAutoMap

```py
>>> from arraymap import FrozenAutoMap
```

`FrozenAutoMap` objects are immutable. They can be constructed from any iterable
of hashable, unique keys.


```py
>>> a = FrozenAutoMap("AAA")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: 'A'
>>> a = FrozenAutoMap("ABC")
>>> a
arraymap.FrozenAutoMap(['A', 'B', 'C'])
```

The values are integers, incrementing according to the order of the original
keys:

```py
>>> a["A"]
0
>>> a["C"]
2
>>> a["X"]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
KeyError: 'X'
```

The full `Mapping` interface is provided:

```py
>>> [*a.keys()]
['A', 'B', 'C']
>>> [*a.values()]
[0, 1, 2]
>>> [*a.items()]
[('A', 0), ('B', 1), ('C', 2)]
>>> a.get("X", 42)
42
>>> "B" in a
True
>>> [*a]
['A', 'B', 'C']
```

They may also be combined with each other using the `|` operator:

```py
>>> b = FrozenAutoMap(range(5))
>>> c = FrozenAutoMap(range(5, 10))
>>> b | c
arraymap.FrozenAutoMap([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> b |= c  # Note that b is reassigned, not mutated!
>>> b
arraymap.FrozenAutoMap([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

### AutoMap

```py
>>> from arraymap import AutoMap
```

Unlike `FrozenAutoMap` objects, `AutoMap` objects can grow; new keys may be
added, but existing ones may not be deleted or changed.

```py
>>> d = AutoMap("ABC")
>>> d
arraymap.AutoMap(['A', 'B', 'C'])
>>> d |= "DEF"  # Here, d *is* mutated!
>>> d
arraymap.AutoMap(['A', 'B', 'C', 'D', 'E', 'F'])
```

They also have `add` and `update` methods for adding new keys:

```py
>>> e = AutoMap(["I", "II", "III"])
>>> e.add("IV")
>>> e
arraymap.AutoMap(['I', 'II', 'III', 'IV'])
>>> e.update(["V", "VI", "VII"])
>>> e
arraymap.AutoMap(['I', 'II', 'III', 'IV', 'V', 'VI', 'VII'])
```

