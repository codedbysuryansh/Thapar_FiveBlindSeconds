# Genuine Linux x86-64 Docker test of champion v1.0.0 (22 Nov 2026)

Machine: cloud VM, Intel Xeon @ 2.10 GHz (2 vCPU), Ubuntu 24.04 host, Docker 29.9.2 (dockerd started as root, overlayfs).
This is a native x86-64 run: no emulation.

Base image. `python:3.12-slim` cannot be pulled here, because Docker Hub, public.ecr.aws and ghcr.io are refused by the
session's egress proxy. Instead, a minimal Ubuntu 24.04 (noble) root filesystem was built with `debootstrap
--variant=minbase` from archive.ubuntu.com (python3 3.12.3, python3-venv, ca-certificates) and loaded with `docker import`.
NumPy 2.5.5 (the version the Phase 4 arm64 container used) was installed offline from the PyPI manylinux_2_28 x86-64 wheel.
The rest is the release Dockerfile unchanged: single-thread environment variables, `COPY main.py`, `ENTRYPOINT ["python",
"/app/main.py"]`. The container ran with `--network none`. See Dockerfile.x86_offline.

What this test does not cover: the official python:3.12-slim base (Debian, Python 3.12.x from python.org). The Phase 4
arm64 container used that image. Python 3.12.3 on glibc 2.39 is the closest x86-64 equivalent available here.

Results
- Output identity: 34/34 seed-42 missions (17 sequences × column orders A and B), run through the container ENTRYPOINT, are
  byte-identical to the macOS arm64 outputs of the same main.py (SHA-256 lists in x86_sha256.txt and out_t2_sha256.txt).
- tools/measure.py inside the container (5 repeats each):

  | Input | own heap (after imports) | ms per IMU-second (min / median) | end-to-end ms per IMU-second |
  |---|---|---|---|
  | MH_01_easy A42 | 2.620 MB | 7.41 / 7.99 | 7.86 |
  | MH_01_easy B42 | 2.621 MB | 7.78 / 9.25 | 9.31 |
  | room2 A42 | 2.438 MB | 7.72 / 8.13 | 7.25 |

  Start-up: python 0.009–0.010 s, + numpy 0.062–0.076 s, + main 0.090–0.108 s.
- New finding about the "+main" window: it is 10.86 MB here, against 3.12 MB on the Mac. The cause is that the container
  mounts the tools read-only, so no `__pycache__` can be written. Importing main.py then compiles the source, and Python's
  compiler peaks at 10.57 MB of traced memory for main.py (measured directly with tracemalloc around compile()). Only
  0.43 MB of code objects are retained. Running `python main.py` always compiles the script, but tracemalloc only sees it
  if tracing is already on at that moment. If a grader imported main.py under tracemalloc with no bytecode cache, it would
  see this transient peak. The own-heap figure (2.62 MB) does not change.
