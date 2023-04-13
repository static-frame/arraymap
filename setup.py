import setuptools
import typing as tp
import site
import os


AM_VERSION = "0.1.2"


with open("README.md") as file:
    LONG_DESCRIPTION = file.read()


def get_ext_dir(*components: tp.Iterable[str]) -> tp.Sequence[str]:
    dirs = []
    for sp in site.getsitepackages():
        fp = os.path.join(sp, *components)
        if os.path.exists(fp):
            dirs.append(fp)
    return dirs


extension = setuptools.Extension(
    "arraymap",
    ["arraymap.c"],
    include_dirs=get_ext_dir("numpy", "core", "include"),
    library_dirs=get_ext_dir("numpy", "core", "lib"),
    define_macros=[("AM_VERSION", AM_VERSION)],
    libraries=["npymath"],  # not including mlib at this time
)


setuptools.setup(
    author="Christopher Ariza, Brandt Bucher",
    version=AM_VERSION,
    description="Dictionary-like lookup from NumPy array values to their integer positions",
    ext_modules=[extension],
    license="MIT",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    name="arraymap",
    python_requires=">=3.7.0",
    url="https://github.com/static-frame/arraymap",
)
