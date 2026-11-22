#!/usr/bin/env python3
"""Memory and runtime of main.py on one input (Phase 4 test T5). Every figure comes from a fresh Python process.

    python tools/measure.py INPUT [--repeats 5] [--json out.json]

Heap (tracemalloc peak, MB = 10^6 bytes), three windows, because the organisers' window is not published:
  after_imports   tracing starts after `import numpy` and `import main`  -> the estimation's own data structures and buffers
  after_numpy     tracing starts after `import numpy`, before `import main` (main's own imports included)
  whole_program   tracing starts before `import numpy` (NumPy's import alone is ~7.5-8 MB of heap)
  interpreter     `python -X tracemalloc`: as whole_program plus the interpreter's start-up allocations
Runtime (wall clock, single thread):
  run             main.main([INPUT, OUTPUT]) inside a process whose imports are done, per IMU-second of input
  startup         interpreter start + `import numpy` + `import main` (= a process that only imports), and its parts
  end_to_end      `python main.py INPUT OUTPUT` as a subprocess
Per-stage time and heap of the after_imports window come from main.PROF."""
import sys, os, json, time, subprocess, platform, tempfile, statistics
HERE = os.path.dirname(os.path.abspath(__file__)); MAIN = os.path.join(os.path.dirname(HERE), "main.py")
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
LOAD = "import importlib.util as _u; _s = _u.spec_from_file_location('main', %r); M = _u.module_from_spec(_s); _s.loader.exec_module(M)" % MAIN
def py(code, *xopt):
    r = subprocess.run([sys.executable, *xopt, "-c", code], capture_output=True, text=True, env=ENV)
    if r.returncode: raise RuntimeError(r.stderr[-2000:])
    return json.loads(r.stdout.strip().splitlines()[-1])
def heap(inp, out, window):
    pre = {"after_imports": "import numpy\n%s\nimport tracemalloc\ntracemalloc.start()\n" % LOAD,
           "after_numpy": "import numpy\nimport tracemalloc\ntracemalloc.start()\n%s\n" % LOAD,
           "whole_program": "import tracemalloc\ntracemalloc.start()\nimport numpy\n%s\n" % LOAD,
           "interpreter": "import tracemalloc\nimport numpy\n%s\n" % LOAD}[window]
    code = pre + "M.PROF = []\nM.main([%r, %r])\nimport json, resource\ncur, peak = tracemalloc.get_traced_memory()\n" % (inp, out)
    code += "st = [(n, p) for n, _t, p in M.PROF]\npk = max([peak] + [p for n, p in st])\n"
    code += "print(json.dumps(dict(peak=pk, stages=st, maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)))"
    return py(code, *(("-X", "tracemalloc") if window == "interpreter" else ()))
def run_time(inp, out):
    code = "import numpy, time, json\n%s\nM.PROF = []\nt0 = time.perf_counter(); M.main([%r, %r]); t1 = time.perf_counter()\n" % (LOAD, inp, out)
    code += "st = [(n, t) for n, t, _p in M.PROF]\nprint(json.dumps(dict(run=t1 - t0, stages=[(st[i][0], st[i][1] - st[i - 1][1]) for i in range(1, len(st))])))"
    return py(code)
def wall(args):
    t0 = time.perf_counter(); r = subprocess.run([sys.executable] + args, capture_output=True, env=ENV); t1 = time.perf_counter()
    if r.returncode: raise RuntimeError(r.stderr[-2000:])
    return t1 - t0
def imu_seconds(inp):
    sys.path.insert(0, HERE); import check_output as CO
    st = [int(x) for x in CO.read_input_stamps(inp)]; sc = CO.unit_scale([str(x) for x in st[:20000]])
    d = sorted(b - a for a, b in zip(st, st[1:]) if b > a); step = d[len(d) // 2] * sc * 1e-9 if d else 0.0
    return (max(st) - min(st)) * sc * 1e-9 + step, len(st)
def machine():
    info = dict(platform=platform.platform(), machine=platform.machine(), python=platform.python_version(), processor=platform.processor())
    info["numpy"] = py("import numpy, json; print(json.dumps(numpy.__version__))")
    try:
        if sys.platform == "darwin": info["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        else:
            for ln in open("/proc/cpuinfo"):
                if ln.startswith("model name"): info["cpu"] = ln.split(":", 1)[1].strip(); break
    except Exception: pass                                              # noqa: BLE001
    return info
def measure(inp, repeats=5):
    out = os.path.join(tempfile.mkdtemp(prefix="fbs_measure_"), "estimated_trajectory.txt"); T, rows = imu_seconds(inp)
    res = dict(input=os.path.abspath(inp), rows=rows, imu_seconds=T, machine=machine(), heap_MB={}, stage_heap_MB={})
    for w in ("after_imports", "after_numpy", "whole_program", "interpreter"):
        h = heap(inp, out, w); res["heap_MB"][w] = h["peak"] / 1e6
        if w == "after_imports": res["stage_heap_MB"] = {n: p / 1e6 for n, p in h["stages"]}; res["maxrss_raw"] = h["maxrss"]
    runs = [run_time(inp, out) for _ in range(repeats)]; rt = [r["run"] for r in runs]
    res["run_s"] = dict(min=min(rt), median=statistics.median(rt), all=rt); res["ms_per_imu_second"] = dict(min=1000 * min(rt) / T, median=1000 * statistics.median(rt) / T)
    best = min(runs, key=lambda r: r["run"]); res["stage_s"] = dict(best["stages"])
    st = {k: [] for k in ("python_only", "import_numpy", "import_main", "end_to_end")}
    for _ in range(repeats):
        st["python_only"].append(wall(["-c", "pass"])); st["import_numpy"].append(wall(["-c", "import numpy"]))
        st["import_main"].append(wall(["-c", "import numpy\n" + LOAD])); st["end_to_end"].append(wall([MAIN, inp, out]))
    res["startup_s"] = {k: dict(min=min(v), median=statistics.median(v)) for k, v in st.items()}
    e2e = res["startup_s"]["end_to_end"]["median"]; res["end_to_end_ms_per_imu_second"] = 1000 * e2e / T
    return res
if __name__ == "__main__":
    a = sys.argv[1:]; rep = 5; js = None
    if "--repeats" in a: i = a.index("--repeats"); rep = int(a[i + 1]); del a[i:i + 2]
    if "--json" in a: i = a.index("--json"); js = a[i + 1]; del a[i:i + 2]
    if len(a) != 1: print(__doc__); sys.exit(2)
    r = measure(a[0], rep); h = r["heap_MB"]; s = r["startup_s"]
    print("%s: %d rows, %.1f s of IMU on %s (%s, Python %s, NumPy %s)" % (os.path.basename(a[0]), r["rows"], r["imu_seconds"], r["machine"].get("cpu", "?"), r["machine"]["platform"], r["machine"]["python"], r["machine"]["numpy"]))
    print("  heap peak MB: after imports %.3f | after numpy %.3f | whole program %.3f | with interpreter start-up %.3f" % (h["after_imports"], h["after_numpy"], h["whole_program"], h["interpreter"]))
    print("  run: %.3f s min, %.3f s median -> %.2f / %.2f ms per IMU-second" % (r["run_s"]["min"], r["run_s"]["median"], r["ms_per_imu_second"]["min"], r["ms_per_imu_second"]["median"]))
    print("  start-up (median s): python %.3f, + import numpy %.3f, + import main %.3f; end to end %.3f s = %.2f ms per IMU-second" % (s["python_only"]["median"], s["import_numpy"]["median"], s["import_main"]["median"], s["end_to_end"]["median"], r["end_to_end_ms_per_imu_second"]))
    print("  stages (s / heap MB): " + ", ".join("%s %.3f/%.2f" % (k, v, r["stage_heap_MB"].get(k, float("nan"))) for k, v in r["stage_s"].items()))
    if js:
        with open(js, "w") as f: json.dump(r, f, indent=1)
