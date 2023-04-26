import os
import sys
import timeit
import typing as tp
from typing import NamedTuple
from itertools import repeat

import arraymap
from arraymap import AutoMap
from arraymap import FrozenAutoMap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())


class PayLoad:
    def __init__(self, array: np.ndarray):
        self.array = array
        self.fama = FrozenAutoMap(self.array)
        self.sel_array = array[(np.arange(len(array)) % 2) == 0]
        self.sel_scalar = list(self.sel_array)
        self.sel_obj = self.sel_array.tolist()


class MapProcessor:
    NAME = ""
    SORT = -1

    def __init__(self, pl: PayLoad):
        self.array = pl.array
        self.fama = pl.fama
        self.sel_array = pl.sel_array
        self.sel_scalar = pl.sel_scalar
        self.sel_obj = pl.sel_obj


# -------------------------------------------------------------------------------
class ListCompAllScalar(MapProcessor):
    NAME = "all: list comp, lookup by scalar"
    SORT = 0

    def __call__(self):
        post = [self.fama[e] for e in self.sel_array]
        assert len(post) == len(self.fama) // 2


class GetAllListObj(MapProcessor):
    NAME = "all: get all, lookup by obj list"
    SORT = 0

    def __call__(self):
        post = self.fama.get_all(self.sel_obj)
        assert len(post) == len(self.fama) // 2


class GetAllListScalar(MapProcessor):
    NAME = "all: get all, lookup by scalar list"
    SORT = 0

    def __call__(self):
        post = self.fama.get_all(self.sel_scalar)
        assert len(post) == len(self.fama) // 2


class GetAllArray(MapProcessor):
    NAME = "all: get all, lookup by array"
    SORT = 0

    def __call__(self):
        post = self.fama.get_all(self.sel_array)
        assert len(post) == len(self.fama) // 2


# -------------------------------------------------------------------------------
class ListCompAnyScalar(MapProcessor):
    NAME = "any: list comp, lookup by scalar"
    SORT = 0

    def __call__(self):
        post = [self.fama[e] for e in self.sel_array if e in self.fama]
        assert len(post) == len(self.fama) // 2


class GetAnyListObj(MapProcessor):
    NAME = "any: get all, lookup by obj list"
    SORT = 0

    def __call__(self):
        post = self.fama.get_any(self.sel_obj)
        assert len(post) == len(self.fama) // 2


class GetAnyListScalar(MapProcessor):
    NAME = "any: get all, lookup by scalar list"
    SORT = 0

    def __call__(self):
        post = self.fama.get_any(self.sel_scalar)
        assert len(post) == len(self.fama) // 2


class GetAnyArray(MapProcessor):
    NAME = "any: get all, lookup by array"
    SORT = 0

    def __call__(self):
        post = self.fama.get_any(self.sel_array)
        assert len(post) == len(self.fama) // 2


# -------------------------------------------------------------------------------
INT_START = 500  # avoid cached ints starting at 256


class FixtureFactory:
    NAME = ""
    SORT = 0
    CACHE = {}  # can be shared for all classes

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        raise NotImplementedError()

    @classmethod
    def get_label_array(cls, size: int) -> tp.Tuple[str, PayLoad]:
        key = (cls, size)
        if key not in cls.CACHE:
            pl = PayLoad(cls.get_array(size))
            cls.CACHE[key] = pl
        return cls.NAME, cls.CACHE[key]


class FFInt64(FixtureFactory):
    NAME = "int64"
    SORT = 0

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        array = np.arange(INT_START, INT_START + size, dtype=np.int64)
        array.flags.writeable = False
        return array


class FFInt32(FixtureFactory):
    NAME = "int32"
    SORT = 1

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        array = np.arange(INT_START, INT_START + size, dtype=np.int32)
        array.flags.writeable = False
        return array


class FFUInt64(FixtureFactory):
    NAME = "uint64"
    SORT = 2

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        array = np.arange(INT_START, INT_START + size, dtype=np.uint64)
        array.flags.writeable = False
        return array


class FFFloat64(FixtureFactory):
    NAME = "float64"
    SORT = 3

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        array = (np.arange(INT_START, INT_START + size) * 0.5).astype(np.float64)
        array.flags.writeable = False
        return array


class FFFloat32(FixtureFactory):
    NAME = "float32"
    SORT = 4

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        array = (np.arange(INT_START, INT_START + size) * 0.5).astype(np.float32)
        array.flags.writeable = False
        return array


class FFString(FixtureFactory):
    NAME = "string"
    SORT = 6

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        array = np.array([hex(e) for e in range(size)])
        array.flags.writeable = False
        return array


class FFString4x(FixtureFactory):
    NAME = "string 4x"
    SORT = 7

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        array = np.array([hex(e) * 4 for e in range(size)])
        array.flags.writeable = False
        return array


class FFBytes(FixtureFactory):
    NAME = "bytes"
    SORT = 8

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        array = np.array([bytes(hex(e), encoding="utf-8") for e in range(size)])
        array.flags.writeable = False
        return array


class FFObject(FixtureFactory):
    NAME = "object"
    SORT = 5

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        ints = np.arange(INT_START, INT_START + size)
        array = ints.astype(object)

        target = 1 == ints % 3
        array[target] = ints[target] * 0.5

        target = 2 == ints % 3
        array[target] = np.array([hex(e) for e in ints[target]])

        array.flags.writeable = False
        return array


def get_versions() -> str:
    import platform

    return f"OS: {platform.system()} / ArrayMap: {arraymap.__version__} / NumPy: {np.__version__}\n"


CLS_FF = (
    FFInt64,
    FFUInt64,
    FFFloat64,
    FFString,
    FFBytes,
)
FF_ORDER = [f.NAME for f in sorted(CLS_FF, key=lambda ff: ff.SORT)]

# -------------------------------------------------------------------------------
NUMBER = 20

from itertools import product


def seconds_to_display(seconds: float) -> str:
    seconds /= NUMBER
    if seconds < 1e-4:
        return f"{seconds * 1e6: .1f} (µs)"
    if seconds < 1e-1:
        return f"{seconds * 1e3: .1f} (ms)"
    return f"{seconds: .1f} (s)"


def plot_performance(frame, suffix: str = ""):
    fixture_total = len(frame["fixture"].unique())
    cat_total = len(frame["size"].unique())
    processor_total = len(frame["cls_processor"].unique())
    fig, axes = plt.subplots(cat_total, fixture_total)

    # cmap = plt.get_cmap('terrain')
    cmap = plt.get_cmap("plasma")
    color = cmap(np.arange(processor_total) / processor_total)

    # category is the size of the array
    for cat_count, (cat_label, cat) in enumerate(frame.groupby("size")):

        # fixture is the data type fixture
        fixture_data = {fix_label: fix for fix_label, fix in cat.groupby("fixture")}
        for fixture_count, fixture_label in enumerate(FF_ORDER):
            fixture = fixture_data[fixture_label]
            ax = axes[cat_count][fixture_count]

            # set order by cls_processor, i.e., the type of test being done
            fixture["sort"] = [f.SORT for f in fixture["cls_processor"]]
            fixture = fixture.sort_values("sort")

            results = fixture["time"].values.tolist()
            names = [cls.NAME for cls in fixture["cls_processor"]]
            # x = np.arange(len(results))
            names_display = names
            post = ax.bar(names_display, results, color=color)

            # density, position = fixture_label.split('-')
            # cat_label is the size of the array
            title = f"{cat_label:.0e}\n{fixture_label}"

            ax.set_title(title, fontsize=6)
            ax.set_box_aspect(0.8)
            time_max = fixture["time"].max()
            ax.set_yticks([0, time_max * 0.5, time_max])
            ax.set_yticklabels(
                [
                    "",
                    seconds_to_display(time_max * 0.5),
                    seconds_to_display(time_max),
                ],
                fontsize=6,
            )
            # ax.set_xticks(x, names_display, rotation='vertical')
            ax.tick_params(
                axis="x",
                which="both",
                bottom=False,
                top=False,
                labelbottom=False,
            )

    fig.set_size_inches(9, 4)  # width, height
    fig.legend(post, names_display, loc="center right", fontsize=6)
    # horizontal, vertical
    fig.text(0.05, 0.96, f"AutoMap {suffix.title()}: {NUMBER} Iterations", fontsize=10)
    fig.text(0.05, 0.90, get_versions(), fontsize=6)

    fp = f"/tmp/arraymap-{suffix}.png"
    plt.subplots_adjust(
        left=0.075,
        bottom=0.05,
        right=0.8,
        top=0.80,
        wspace=0.6,  # width
        hspace=0.2,
    )
    # plt.rcParams.update({'font.size': 22})
    print(fp)
    plt.savefig(fp, dpi=300)

    if sys.platform.startswith("linux"):
        os.system(f"eog {fp}&")
    else:
        os.system(f"open {fp}")


def run_test(processors, suffix):
    records = []
    for size in (10_000, 100_000, 1_000_000):
        for ff in CLS_FF:
            fixture_label, fixture = ff.get_label_array(size)
            for cls in processors:
                runner = cls(fixture)

                record = [cls, NUMBER, fixture_label, size]
                print(record)
                try:
                    result = timeit.timeit(f"runner()", globals=locals(), number=NUMBER)
                except OSError:
                    result = np.nan
                finally:
                    pass
                record.append(result)
                records.append(record)

    f = pd.DataFrame.from_records(
        records, columns=("cls_processor", "number", "fixture", "size", "time")
    )
    print(f)
    plot_performance(f, suffix)


if __name__ == "__main__":

    cls_instantiate = (
        ListCompAllScalar,
        GetAllListObj,
        GetAllListScalar,
        GetAllArray,
        ListCompAnyScalar,
        GetAnyListObj,
        GetAnyListScalar,
        GetAnyArray,
    )

    run_test(cls_instantiate, "get-all-any")
