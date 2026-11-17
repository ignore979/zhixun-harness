"""Auditable industrial model continuation."""

import os

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

__version__ = "0.1.0"
