"""an03: cProfile of the champion on one rotor and one hand-held input (diagnostic for the runtime workstream)."""
import os, sys, cProfile, pstats, io, importlib.util, tempfile
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[k] = "1"
import numpy as np
ROOT = "~/SamsungPRISM_FiveBlindSeconds"; path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "champion/v1.0.0/release/main.py")
spec = importlib.util.spec_from_file_location("m", path); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
for tag in ("MH_01_easy_A42", "room2_A42"):
    inp = os.path.join(ROOT, "opt/cache/missions", tag + ".zip"); out = os.path.join(tempfile.mkdtemp(), "est.txt")
    M.run(inp, out)                                                           # warm-up
    pr = cProfile.Profile(); pr.enable(); M.run(inp, out); pr.disable()
    s = io.StringIO(); ps = pstats.Stats(pr, stream=s).sort_stats("tottime"); ps.print_stats(28); print("=" * 30, tag); print(s.getvalue()[:6000])
    s = io.StringIO(); ps = pstats.Stats(pr, stream=s).sort_stats("cumulative"); ps.print_stats(30); print(s.getvalue()[:5000])
