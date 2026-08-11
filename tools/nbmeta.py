"""Notebook metadata, defined once.

The kernel a notebook **declares** and the kernel `tools/build.py` **executes**
it with have to be the same name. They were not: every generator emitted
`"python3"` while build.py passed `--ExecutePreprocessor.kernel_name=quth`, the
kernel the README tells you to register from the repo venv. Anyone running
`jupyter nbconvert --execute` (or hitting Run All in Jupyter) without build.py's
explicit override therefore got whatever `python3` resolves to on their machine
-- typically the system interpreter, where `qviz` is not installed.

Both sides now read KERNEL from here. Override with QUTH_KERNEL if you register
the venv kernel under a different name.
"""

from __future__ import annotations

import os

KERNEL = os.environ.get("QUTH_KERNEL", "quth")
DISPLAY_NAME = "Python 3 (quth)"
PYTHON_VERSION = "3.13.7"


def notebook_metadata():
    """The `nb.metadata` block every generator ends with."""
    return {
        "kernelspec": {"display_name": DISPLAY_NAME, "language": "python",
                       "name": KERNEL},
        "language_info": {"name": "python", "version": PYTHON_VERSION},
    }
