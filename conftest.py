import os
import sys

# On Windows, numpy (MKL) and torch (libiomp) can both load an OpenMP runtime in
# one process (the full test suite imports both), which aborts with "OMP Error #15".
# This is the standard guard; our CPU numerics are unaffected by it.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# make the repo root importable so `import rwvae` works without installation,
# and the scripts/ dir importable so tests can drive the eval/demo entry points
_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "scripts"))
