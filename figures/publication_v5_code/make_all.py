#!/usr/bin/env python3
"""Generate publication_v5 main and supplementary figures."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent

SCRIPTS = [
    "figure1_cohort_landscape.py",
    "figure2_study_volume_stage.py",
    "figure3_adjusted_and_matched.py",
    "figure4_portability.py",
    "figure5_colocalization_by_pool.py",
    "supplementary_figures.py",
]


def main():
    for script in SCRIPTS:
        path = HERE / script

        print("=" * 96)
        print("RUNNING", path.name)
        print("=" * 96)

        subprocess.run(
            [sys.executable, str(path)],
            cwd=HERE,
            check=True,
        )

    print(
        "All publication_v5 figures completed."
    )


if __name__ == "__main__":
    main()
