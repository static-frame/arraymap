import functools
import sys

import invoke

run = functools.partial(invoke.Context.run, echo=True, pty=True)


@invoke.task
def install(context):
    # type: (invoke.Context) -> None
    run(context, f"{sys.executable} -m pip install --upgrade pip")
    run(context, f"{sys.executable} -m pip install --upgrade -r requirements.txt")


@invoke.task()
def clean(context):
    # type: (invoke.Context) -> None
    # run(context, f"{sys.executable} setup.py develop --uninstall")
    run(context, f"{sys.executable} -m pip uninstall --yes arraymap")

    for artifact in ("*.egg-info", "*.so", "build", "dist"):
        run(context, f"rm -rf {artifact}")
    run(context, f"{sys.executable} -m black .")


@invoke.task(clean)
def build(context):
    # type: (invoke.Context) -> None
    # run(context, f"{sys.executable} setup.py develop")
    run(context, f"{sys.executable} -m pip -v install .")


@invoke.task(build)
def test(context):
    # type: (invoke.Context) -> None
    run(context, f"{sys.executable} -m pytest -v")
