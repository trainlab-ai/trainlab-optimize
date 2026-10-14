"""Draw the stratified random sample: bench/repos.json (ordered draw per category, with spares).

    python bench/sample.py [--seed 20261014]

Each category lists its quota of picks followed by the remaining candidates in random order;
the runner uses spares, in order, to replace repos that fail to clone.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from pool import POOL

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261014)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = {"seed": args.seed, "categories": {}}
    for cat, spec in POOL.items():
        order = list(spec["repos"])
        rng.shuffle(order)
        out["categories"][cat] = {"quota": spec["quota"], "llm": spec["llm"], "draw": order}
    (HERE / "repos.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {HERE / 'repos.json'}: {sum(c['quota'] for c in out['categories'].values())} picks, "
          f"{sum(len(c['draw']) for c in out['categories'].values())} candidates")


if __name__ == "__main__":
    main()
