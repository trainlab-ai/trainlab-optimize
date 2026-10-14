"""Shared prelude: instrument first, the way an app would at startup."""
import os

import trainlab_optimize as trainlab

trainlab.instrument(path=os.environ["TRAINLAB_TRACES"], project_root=os.environ["FW_ROOT"])
BASE = os.environ["FAKE_BASE"]
