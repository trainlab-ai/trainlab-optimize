"""Stage 1 of the 100-repo experiment test: one workspace per frozen repo.

For each repo in bench/repos.lock.json: fetch the pinned commit, scan it, import the same
mock usage the Profile 1 suite uses, and keep only the workspace (scan.json, usage.json).
The clone is deleted afterwards. Experiments (stage 2) need no source code.

    python bench/experiments/prepare.py WORKDIR
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH))

from mock_usage import FORMATS, generate  # noqa: E402
from run import clone, slug  # noqa: E402

from trainlab_optimize.importers.auto import read_any  # noqa: E402
from trainlab_optimize.scanner import scan  # noqa: E402
from trainlab_optimize.workspace import Workspace  # noqa: E402


def prepare(item: dict, i: int, work: Path) -> dict:
    full, sha, cat = item["repo"], item["sha"], item["category"]
    out = work / "ws" / slug(full)
    done = out / "meta.json"
    if done.exists() and json.loads(done.read_text()).get("ok"):
        return json.loads(done.read_text())
    src = work / "src" / slug(full)
    shutil.rmtree(src, ignore_errors=True)
    src.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "mock").mkdir(exist_ok=True)
    ok, err = clone(full, src, sha=sha)
    meta = {"repo": full, "category": cat, "sha": sha, "ok": ok, "error": err}
    try:
        if ok:
            res = scan(src)
            ws = Workspace(out)
            ws.save_scan(res)
            fmt = FORMATS[i % len(FORMATS)]
            exp = generate(full, res.call_sites, out / "mock", fmt)
            records, _ = read_any(exp["path"], column_map=exp["column_map"])
            ws.add_usage(records)
            meta.update(sites=len(res.call_sites), monthly_mock=round(exp["total_cost"], 2))
    except Exception as exc:  # noqa: BLE001
        meta.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(src, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(meta))
    return meta


if __name__ == "__main__":
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    lock = json.loads((BENCH / "repos.lock.json").read_text())["repos"]
    with cf.ThreadPoolExecutor(4) as ex:
        for m in ex.map(lambda a: prepare(a[1], a[0], work), enumerate(lock)):
            print(("ok  " if m["ok"] else "FAIL"), m["repo"], m.get("sites"), m.get("error", "")[:80], flush=True)
