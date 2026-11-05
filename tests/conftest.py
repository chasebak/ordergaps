"""Shared fixtures.

The frozen evaluation matrix takes several seconds, and both the acceptance
band and the README drift test need it, so it is built once per test session
instead of once per test file.
"""

import pytest

from ordergaps.evaluation.experiments import run_matrix
from ordergaps.evaluation.run_all import FROZEN_SEEDS, SHARES, frozen_base_config


@pytest.fixture(scope="session")
def frozen_records():
    records = run_matrix(frozen_base_config(), SHARES, FROZEN_SEEDS)
    return records
