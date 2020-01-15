from glob import glob
from os import environ, remove, replace
from sys import executable, platform

from invoke import task


@task
def clean(context) -> None:
    context.run(f"{executable} setup.py develop --uninstall")
    for artifact in ("*.egg-info", "*.so", "build", "dist"):
        context.run("rm -rf {artifact}".format(artifact=artifact))
    context.run("black .")


@task(clean)
def build(context) -> None:
    context.run("pip install -r requirements.txt")
    context.run(
        f"{executable} setup.py develop sdist bdist_wheel",
        env={"CPPFLAGS": "-Werror -Wno-deprecated-declarations"},
        replace_env=False,
    )

    WHEELS = glob("dist/*.whl")
    assert WHEELS, "No wheels in dist!"
    print("Before:", *WHEELS, sep="\n - ")
    if platform == "linux":
        # We're typically eligible for manylinux1... or at least manylinux2010.
        # This will remove the wheel if it was unchanged... but that will cause
        # our assert to fail later, which is what we want!
        for wheel in WHEELS:
            context.run(f"auditwheel repair {wheel} -w dist")
            remove(wheel)
    elif platform == "darwin":
        # We lie here, and say our 10.9 64-bit build is a 10.6 32/64-bit one.
        # This is because pip is conservative in what wheels it will use, but
        # Python installations are EXTREMELY liberal in their macOS support.
        # A typical user may be running a 32/64 Python built for 10.6.
        # In reality, we shouldn't worry about supporting 32-bit Snow Leopard.
        for wheel in WHEELS:
            fake = wheel.replace("macosx_10_9_x86_64", "macosx_10_6_intel")
            replace(wheel, fake)
            assert (
                wheel != fake or "TRAVIS" not in environ
            ), "We expected a macOS 10.9 x86_64 build!"
    # Windows is fine.
    FIXED = glob("dist/*.whl")
    print("After:", *FIXED, sep="\n - ")
    assert len(WHEELS) == len(FIXED), "We gained or lost a wheel!"

    context.run("twine check dist/*")
    context.run("pip install --force-reinstall --no-cache-dir .")
    context.run("pip install --force-reinstall --no-cache-dir dist/*.tar.gz")
    context.run("pip install --force-reinstall --no-cache-dir dist/*.whl")


@task(build)
def test(context) -> None:
    context.run("pytest -v")


@task(test)
def performance(context) -> None:
    context.run("{} performance.py".format(executable))


@task(test)
def release(context) -> None:
    context.run("twine upload --skip-existing dist/*")
