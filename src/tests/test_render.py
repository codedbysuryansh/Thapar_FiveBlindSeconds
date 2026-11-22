"""Renderer == %-formatting: 10 million random rows (realistic and adversarial magnitudes, exact rounding ties) + edge values."""
import sys, os, numpy as np, time, importlib.util
spec = importlib.util.spec_from_file_location("main", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "main.py")); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
rng = np.random.default_rng(1); bad = 0; tot = 0; t0 = time.time()
def ref(tn, P, Q): return M._fmt_py(tn, P, Q)
for rep in range(40):
    n = 250000
    P = np.concatenate([rng.normal(0, 10, (n // 2, 3)), rng.uniform(-1e3, 1e3, (n // 2, 3))]) * rng.choice([1, 1e-3, 1e-6], (n, 1))
    Q = rng.normal(size=(n, 4)); Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    if rep % 4 == 0:                                                    # exact decimal ties and their binary neighbours
        k = rng.integers(-10 ** 9, 10 ** 9, (n, 3)); P = (k + 0.5) / 1e6; P[: n // 3] = np.nextafter(P[: n // 3], np.inf); P[n // 3: 2 * n // 3] = np.nextafter(P[n // 3: 2 * n // 3], -np.inf)
        kq = rng.integers(-10 ** 7, 10 ** 7, (n, 4)); Q = (kq + 0.5) / 1e7
    tn = rng.integers(0, 2 * 10 ** 18, n)
    for s in range(0, n, 4096):
        a = M.render(tn[s:s + 4096], P[s:s + 4096], Q[s:s + 4096]); b = ref(tn[s:s + 4096], P[s:s + 4096], Q[s:s + 4096])
        if a != b:
            la, lb = a.split(b"\n"), b.split(b"\n"); bad += sum(x != y for x, y in zip(la, lb))
        tot += min(4096, n - s)
E = np.array([[0.0, -0.0, 5e-7, -5e-7, 4.9999999e-7, 999999999.9999994, -999999999.9999994, 1e-300]]).T
Pe = np.repeat(E, 3, 1); Qe = np.repeat(E, 4, 1); te = np.array([0, 1, 10 ** 9 - 1, 10 ** 9, 1403636579758555392, 2 ** 63 - 1, 999, 5])
eb = M.render(te, Pe, Qe) != ref(te, Pe, Qe)
print("renderer vs %%-formatting: %d random rows, %d mismatching; edge cases mismatch: %s; %.0f s -> %s" % (tot, bad, eb, time.time() - t0, "PASS" if not bad and not eb else "FAIL"))
sys.exit(0 if not bad and not eb else 1)
