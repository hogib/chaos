"""Convenience shim so ``python src/main.py`` still starts the pipeline.

The implementation lives in :mod:`chaos.pipeline` so that the installed
``chaos`` console script runs the same code.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chaos.pipeline import main  # noqa: E402

if __name__ == "__main__":
    main()
