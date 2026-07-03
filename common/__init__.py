"""Dependency-light shared helpers for the staged reconstruction pipeline.

Hard rule (Section 3.5 of the build specification): everything in this package
depends on nothing beyond the Python standard library and ``numpy``. This is what
lets every stage install and import it into an otherwise-conflicting environment
via ``pip install -e ./common``. Anything heavier belongs inside the stage that
needs it, never here.
"""

__version__ = "0.1.0"
