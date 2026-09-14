"""Every tracked module must import from an empty directory, silently.

Written after `train.py` shipped with exploration code at module level: an
`import` that read `data/clips` and printed four lines. `data/` is gitignored,
so on any clone `import train` raised `FileNotFoundError` - and nothing in the
suite noticed, because the suite runs where the data happens to exist.

The same failure class has now appeared twice. The first time a tracked module
imported an untracked one; this time a tracked module read an untracked
directory. Both are invisible on the machine that wrote them and fatal on every
other machine, which is the definition of a bug a test has to catch.

Two properties, both checked by importing in a subprocess whose working
directory is empty:

1. **It must succeed.** Anything an import reaches for - a data directory, a
   config file, a sibling module that is not in the repo - fails here.
2. **It must be silent.** A module that prints on import is a module doing work
   on import. The output is the symptom; the work is the defect.

Import time is measured too, and kept well under a second. A module that loads
a dataset quietly would pass the two checks above and still be wrong.
"""

import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

def tracked_modules():
    """Top-level modules git knows about - which is exactly what a clone gets.

    Asking git rather than globbing the directory is the whole point: an
    untracked scratch file is not part of the repo and does not need to import
    anywhere, while a tracked one does. No exclusion list to keep in step, and
    nothing can be exempted by being named a particular way.
    """
    listed = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=REPO, capture_output=True, text=True, timeout=30,
    )
    return sorted(
        Path(line).stem
        for line in listed.stdout.splitlines()
        if "/" not in line
    )


MODULES = tracked_modules()


def import_from_empty_dir(module, tmp_path):
    """Import `module` in a fresh interpreter whose cwd holds nothing."""
    started = time.monotonic()
    finished = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=tmp_path,
        env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    return finished, time.monotonic() - started


def test_there_is_something_to_check():
    """Guards the glob: a typo in the exclude list would empty this suite."""
    assert set(MODULES) >= {"boids", "generate", "train", "windows"}, MODULES


@pytest.mark.parametrize("module", MODULES)
def test_the_module_imports_with_no_data_directory(module, tmp_path):
    """A clone has the code and none of the generated data."""
    finished, _ = import_from_empty_dir(module, tmp_path)

    assert finished.returncode == 0, (
        f"`import {module}` failed outside the repo directory. Anyone who "
        f"clones this repo gets:\n{finished.stderr}")


@pytest.mark.parametrize("module", MODULES)
def test_the_module_prints_nothing_on_import(module, tmp_path):
    """Printing on import means working on import."""
    finished, _ = import_from_empty_dir(module, tmp_path)

    assert finished.stdout == "", (
        f"`import {module}` printed:\n{finished.stdout}"
        "Module level belongs to definitions. Work belongs in a function.")
    assert finished.stderr == "", (
        f"`import {module}` wrote to stderr:\n{finished.stderr}")


@pytest.mark.parametrize("module", MODULES)
def test_the_module_imports_quickly(module, tmp_path):
    """Silent work is still work - this is the check the other two cannot make."""
    _, elapsed = import_from_empty_dir(module, tmp_path)

    assert elapsed < 5.0, (
        f"`import {module}` took {elapsed:.1f}s. That is not a module being "
        "defined, it is a module doing something.")
