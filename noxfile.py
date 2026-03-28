import logging
import pathlib
import shutil

import nox
import nox_uv

logger = logging.getLogger(__name__)

nox.options.sessions = []
nox.options.default_venv_backend = "uv"


@nox_uv.session(
    venv_backend="uv",
    uv_groups=["dev"],
    uv_quiet=True,
)
def version(session: nox.Session) -> None:
    """Prints version."""
    session.run("python", "-c", "import prob_conf_mat as pcm;print(pcm.__version__)")


@nox_uv.session(
    venv_backend="uv",
    uv_quiet=True,
)
def install(session: nox.Session) -> None:
    """Prints version."""
    session.run("uv", "sync", "--no-dev", "--refresh", "--reinstall")


@nox_uv.session(
    venv_backend="uv",
    uv_quiet=True,
)
def clean(session: nox.Session) -> None:  # noqa: ARG001  # pyright: ignore[reportUnusedParameter]
    """Clean up caches and build artifacts."""

    def clean_cache_dir(file_path: str) -> None:

        fp = pathlib.Path(file_path)
        logger.info(f"{fp.resolve()}")
        if fp.exists():
            logger.info("\tFound")

            while True:
                answer = input("\tRemove? [Y/N]").lower()

                if answer == "y" or answer == "yes":  # noqa: PLR1714
                    if fp.is_dir():
                        shutil.rmtree(fp)
                    else:
                        fp.unlink()

                    logger.info("\tRemoved")
                    break

                if answer == "n" or answer == "no":  # noqa: PLR1714
                    logger.info("\tNot removing")
                    break

                logger.info(f"\tCould not parse: {answer}")
        else:
            logger.info("\tDid not find")

    clean_cache_dir(file_path="__pycache__")
    clean_cache_dir(file_path=".cache")
    clean_cache_dir(file_path=".coverage")
    clean_cache_dir(file_path=".nox")
    clean_cache_dir(file_path=".pytest_cache")
    clean_cache_dir(file_path=".ruff_cache")
    clean_cache_dir(file_path="dist")
    clean_cache_dir(file_path="site")

    # session.run("python", "-c", "import shutil;shutil.rmtree('.pytest_cache')")
    # session.run("python", "-c", "import shutil;shutil.rmtree('.nox')")
    # session.run("python", "-c", "import shutil;shutil.rmtree('.venv')")


@nox_uv.session(
    venv_backend="uv",
    uv_groups=["dev"],
    uv_quiet=True,
    python=["3.11", "3.12", "3.13", "3.14"],
)
def test(session: nox.Session) -> None:
    """Runs all tests."""
    session.run("pytest")


@nox_uv.session(
    venv_backend="uv",
    uv_groups=["dev"],
    uv_quiet=True,
)
def test_coverage(session: nox.Session) -> None:
    """Checks test coverage."""
    session.run("coverage", "run", "-m", "pytest")
    session.run("coverage", "html")


@nox_uv.session(
    venv_backend="uv",
    uv_groups=["dev"],
    uv_quiet=True,
)
def lint(session: nox.Session) -> None:
    """Run linting."""
    session.run(
        "ruff",
        "check",
        "./src/prob_conf_mat",
        "--fix",
        "--show-fixes",
        "--target-version",
        "py311",
    )


@nox_uv.session(
    venv_backend="uv",
    uv_groups=["dev"],
    uv_quiet=True,
)
def type(session: nox.Session) -> None:  # noqa: A001
    """Run static typechecking."""
    session.run("basedpyright")


@nox_uv.session(
    venv_backend="uv",
    uv_groups=["dev"],
    uv_quiet=True,
)
def commit(session: nox.Session) -> None:
    """Run pre-commit checks."""
    session.run("prek", "run")


if __name__ == "__main__":
    nox.main()
