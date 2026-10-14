"""Create one uv venv per corpus project: its packages + trainlab-optimize (from this checkout)."""
import concurrent.futures as cf
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ENVS = Path(sys.argv[1])
ENVS.mkdir(parents=True, exist_ok=True)
only = set(sys.argv[2:])
corpus = json.loads((HERE / "corpus.json").read_text())


def install(name):
    env = ENVS / name.replace(":", "_")
    log = ENVS / f"{env.name}.install.log"
    with open(log, "w") as fh:
        r = subprocess.run(["uv", "venv", "-q", "--clear", "--python", corpus[name].get("python", "3.12"), str(env)], stdout=fh, stderr=fh)
        if r.returncode == 0:
            r = subprocess.run(["uv", "pip", "install", "-q", "--python", str(env / "bin" / "python"),
                                str(ROOT), *corpus[name]["pkgs"]], stdout=fh, stderr=fh, timeout=1200)
    return name, r.returncode


with cf.ThreadPoolExecutor(8) as ex:
    for name, rc in ex.map(install, [n for n in corpus if not only or n in only]):
        print(("ok  " if rc == 0 else "FAIL"), name, flush=True)
