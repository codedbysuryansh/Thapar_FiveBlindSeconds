#!/usr/bin/env python3
"""Phase 4 test T6 (cold start): a fresh environment that contains only what a user would download - main.py and
requirements.txt - plus NumPy from PyPI, then the README command on a zipped official-format input, then the T1 contract.

    python p4_t6_cold_start.py INPUT.zip REFERENCE_OUTPUT.txt LABEL=PYTHON=NUMPY_SPEC [LABEL=PYTHON=NUMPY_SPEC ...] [--json out.json]

For each LABEL: an empty temporary folder, `PYTHON -m venv` (no system site-packages), `pip install NUMPY_SPEC` (or `-r
requirements.txt` when NUMPY_SPEC is `-r`), `python main.py INPUT estimated_trajectory.txt` inside that folder, then
tools/check_output.py on the result and a byte comparison with REFERENCE_OUTPUT (the same input run by the development
environment). The organisers' Dockerfile route (python:3.12-slim + pip install numpy) is the same sequence inside a container."""
import os, sys, json, time, shutil, subprocess, tempfile, platform
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REL = os.path.join(ROOT, "release"); sys.path.insert(0, os.path.join(REL, "tools")); import check_output as CO   # noqa: E402
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
def run(cmd, cwd=None):
    t0 = time.perf_counter(); r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=ENV); return r, time.perf_counter() - t0
def one(inp, ref, label, py, spec):
    work = tempfile.mkdtemp(prefix="fbs_t6_%s_" % label); out = dict(label=label, python=py, numpy_spec=spec)
    try:
        for f in ("main.py", "requirements.txt"): shutil.copy(os.path.join(REL, f), os.path.join(work, f))
        shutil.copy(inp, os.path.join(work, os.path.basename(inp)))
        r, dt = run([py, "-m", "venv", os.path.join(work, "venv")]); out["venv_s"] = dt
        if r.returncode: out["error"] = "venv: " + r.stderr[-400:]; return out
        vpy = os.path.join(work, "venv", "bin", "python")
        pip = [vpy, "-m", "pip", "install", "-q", "--disable-pip-version-check"] + (["-r", "requirements.txt"] if spec == "-r" else [spec])
        r, dt = run(pip, cwd=work); out["pip_s"] = dt
        if r.returncode: out["error"] = "pip: " + (r.stderr or r.stdout)[-600:]; return out
        r, _ = run([vpy, "-c", "import sys, numpy; print(sys.version.split()[0], numpy.__version__)"]); out["versions"] = r.stdout.strip()
        r, dt = run([vpy, "-m", "pip", "list", "--format=freeze", "--disable-pip-version-check"]); out["installed"] = r.stdout.split()
        r, dt = run([vpy, "main.py", os.path.basename(inp), "estimated_trajectory.txt"], cwd=work); out["run_s"] = dt; out["exit"] = r.returncode
        out["stdout"] = r.stdout.strip()[-200:]; out["stderr"] = r.stderr.strip()[-400:]
        est = os.path.join(work, "estimated_trajectory.txt")
        if not os.path.exists(est): out["error"] = "no output file"; return out
        c = CO.check(inp, est); out["T1"] = c
        out["identical_to_reference"] = open(est, "rb").read() == open(ref, "rb").read() if ref else None
        if not out["identical_to_reference"] and ref:                   # quantify any difference (numerics of other NumPy builds)
            a = [ln.split() for ln in open(est)]; b = [ln.split() for ln in open(ref)]
            out["rows_differing"] = sum(x != y for x, y in zip(a, b))
            out["max_abs_diff"] = max((abs(float(u) - float(v)) for x, y in zip(a, b) if x != y for u, v in zip(x[1:], y[1:])), default=0.0)
        out["pass"] = bool(c.get("ok")) and out["exit"] == 0
        return out
    finally: shutil.rmtree(work, ignore_errors=True)
if __name__ == "__main__":
    a = sys.argv[1:]; js = None
    if "--json" in a: i = a.index("--json"); js = a[i + 1]; del a[i:i + 2]
    inp, ref, specs = a[0], a[1], a[2:]; res = []
    for s in specs:
        label, py, spec = s.split("=", 2); r = one(inp, ref, label, py, spec); res.append(r)
        print("%-14s %-22s pip %5.1f s  run %5.2f s  exit %s  T1 %s  identical %s%s%s" % (label, r.get("versions", "?"), r.get("pip_s", float("nan")), r.get("run_s", float("nan")),
              r.get("exit"), "PASS" if r.get("T1", {}).get("ok") else "FAIL", r.get("identical_to_reference"),
              ("  (%d rows differ, max %.1e)" % (r["rows_differing"], r["max_abs_diff"])) if "rows_differing" in r else "", ("  ERROR " + r["error"]) if "error" in r else ""), flush=True)
    ok = all(r.get("pass") for r in res); print("T6: %s (%d environments, host %s %s)" % ("PASS" if ok else "FAIL", len(res), platform.system(), platform.machine()))
    if js: json.dump(dict(input=inp, host=platform.platform(), results=res, pass_=ok), open(js, "w"), indent=1)
    sys.exit(0 if ok else 1)
