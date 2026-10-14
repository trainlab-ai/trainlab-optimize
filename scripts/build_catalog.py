"""Rebuild the bundled price catalog (maintainers). Users run `trainlab prices update`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trainlab_optimize.catalog.build import main  # noqa: E402

if __name__ == "__main__":
    main()
