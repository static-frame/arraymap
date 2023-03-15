import typing
import pytest
import numpy as np

from automap import AutoMap
from automap import FrozenAutoMap

# from automap import NonUniqueError


def test_am_extend():
    am1 = AutoMap(("a", "b"))
    am2 = am1 | AutoMap(("c", "d"))
    assert list(am2.keys()) == ["a", "b", "c", "d"]


def test_am_add():
    a = AutoMap()
    for l, key in enumerate(["a", "b", "c", "d"]):
        assert a.add(key) is None
        assert len(a) == l + 1
        assert a[key] == l


def test_fam_contains():
    x = []
    fam = FrozenAutoMap(("a", "b", "c"))
    assert (x in fam.values()) == False
    # NOTE: exercise x to force seg fault
    assert len(x) == 0


def test_fam_constructor_array_a1():
    a1 = np.array((10, 20, 30), dtype=np.int64)
    with pytest.raises(TypeError):
        fam = FrozenAutoMap(a1)


def test_fam_constructor_array_a2():
    a1 = np.array((10, 20, 30), dtype=np.int32)
    with pytest.raises(TypeError):
        fam = FrozenAutoMap(a1)


# def test_fam_constructor_array_a3():
#     a1 = np.array(("a", "bb", "ccc"))
#     with pytest.raises(TypeError):
#         fam = FrozenAutoMap(a1)


def test_fam_constructor_array_b():
    a1 = np.array(("2022-01", "2023-05"), dtype=np.datetime64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert fam[np.datetime64("2023-05")] == 1
    # assert np.datetime64('2022-05') in a1


def test_fam_constructor_array_c():
    a1 = np.array((10, 20, 30, 40), dtype=np.int64).reshape(2, 2)
    a1.flags.writeable = False
    with pytest.raises(TypeError):
        fam = FrozenAutoMap(a1)


def test_fam_array_len_a():
    a1 = np.array((10, 20, 30, 40), dtype=np.int64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert len(fam) == 4


def test_fam_array_len_b():
    a1 = np.array((10, 20, 30, 40), dtype=np.int64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert fam[10] == 0
    assert fam[20] == 1
    assert fam[30] == 2
    assert fam[40] == 3


# ------------------------------------------------------------------------------


def test_fam_array_get_a():
    a1 = np.array((1, 100, 300, 4000), dtype=np.int64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    assert fam.get("f") is None
    assert fam.get(1) == 0
    assert fam.get(True) == 0
    assert fam.get(a1[2]) == 2
    assert fam.get(1.0) == 0


def test_fam_array_get_b():
    a1 = np.array((1, 100, 300, 4000), dtype=np.int32)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    assert fam.get("f") is None
    assert fam.get(1) == 0
    assert fam.get(True) == 0
    assert fam.get(a1[2]) == 2
    assert fam.get(1.0) == 0
    assert fam.get(1.1) is None


def test_fam_array_get_c():
    a1 = np.array((1, 5, 10, 20), dtype=np.int16)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    assert fam.get("f") is None
    assert fam.get(1) == 0
    assert fam.get(True) == 0
    assert fam.get(a1[2]) == 2
    assert fam.get(20.0) == 3


def test_fam_array_get_d():
    a1 = np.array((1, 5, 10, 20), dtype=np.int8)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    assert fam.get("f") is None
    assert fam.get(1) == 0
    assert fam.get(True) == 0
    assert fam.get(a1[2]) == 2
    assert fam.get(20.0) == 3
    assert fam.get(20.1) is None


# ------------------------------------------------------------------------------


def test_fam_array_values_a():
    a1 = np.array((10, 20, 30, 40), dtype=np.int64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert list(fam.values()) == [0, 1, 2, 3]


def test_fam_array_keys_a():
    a1 = np.array((10, 20, 30, 40), dtype=np.int64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert list(fam.keys()) == [10, 20, 30, 40]


def test_fam_array_items_a():
    a1 = np.array((10, 20, 30, 40), dtype=np.int64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert list(fam.items()) == [(10, 0), (20, 1), (30, 2), (40, 3)]


def test_fam_array_values_b():
    a1 = np.array(("a", "b", "c", "d"))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert list(fam.values()) == [0, 1, 2, 3]


def test_fam_array_keys_b():
    a1 = np.array(("a", "b", "c", "d"))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert list(fam.keys()) == ["a", "b", "c", "d"]


def test_fam_array_items_b():
    a1 = np.array(("a", "b", "c", "d"))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert list(fam.items()) == [("a", 0), ("b", 1), ("c", 2), ("d", 3)]


def test_fam_array_items_c():
    a1 = np.array(("a", "b", "c"))
    a1.flags.writeable = False
    fam1 = FrozenAutoMap(a1)

    fam2 = FrozenAutoMap(fam1)
    assert list(fam2.items()) == [("a", 0), ("b", 1), ("c", 2)]
    assert list(fam1.items()) == [("a", 0), ("b", 1), ("c", 2)]


def test_am_array_constructor_a():
    a1 = np.array(("a", "b", "c"))
    a1.flags.writeable = False
    am1 = AutoMap(a1)


def test_am_array_constructor_b():
    a1 = np.array(("2022-01", "2023-05"), dtype=np.datetime64)
    a1.flags.writeable = False
    am1 = AutoMap(a1)
    assert am1[np.datetime64("2023-05")] == 1


def test_am_array_constructor_c():
    a1 = np.array((10, 20, 30, 40), dtype=np.int64)
    a1.flags.writeable = False
    am = AutoMap(a1)
    am.update((60, 80))
    am.add(90)
    assert list(am.keys()) == [10, 20, 30, 40, 60, 80, 90]
