"""Stage-time comparison of two main.py builds (best of N runs per stage), same process, alternating."""
import sys, time, importlib.util
def load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
a, b, path = sys.argv[1], sys.argv[2], sys.argv[3]; reps = int(sys.argv[4]) if len(sys.argv) > 4 else 5
mods = [("A", load("A", a)), ("B", load("B", b))]; best = {}; tot = {}
for rep in range(reps):
    for nm, M in mods:
        M.PROF = []; t0 = time.perf_counter(); M.run(path, "/tmp/stagecmp_%s.txt" % nm); dt = time.perf_counter() - t0
        P = M.PROF; st = {P[i][0]: P[i][1] - P[i - 1][1] for i in range(1, len(P))}; bb = best.setdefault(nm, {})
        for k, v in st.items(): bb[k] = min(bb.get(k, 1e9), v)
        tot[nm] = min(tot.get(nm, 1e9), dt)
same = open("/tmp/stagecmp_A.txt", "rb").read() == open("/tmp/stagecmp_B.txt", "rb").read()
print("%s: outputs identical %s; best total A %.3f s, B %.3f s (%+.1f %%)" % (path.split("/")[-1], same, tot["A"], tot["B"], 100 * (tot["B"] / tot["A"] - 1)))
for k in best["A"]: print("   %-16s A %.4f  B %.4f  %+.4f" % (k, best["A"][k], best["B"].get(k, float("nan")), best["B"].get(k, 0) - best["A"][k]))
