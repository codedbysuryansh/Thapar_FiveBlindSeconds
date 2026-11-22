# Champion v1.1.0 (immutable)

Promoted 22 Nov 2026 by the Phase 5 optimisation programme (release-candidate build RC-1.1.0c = research candidate
X110d). This folder is read-only and carries the macOS user-immutable flag. Nothing in it is ever edited. The previous
champion, `champion/v1.0.0/`, stays untouched as the fallback.

## What it is

- `release/` is the release candidate folder `rc/` at promotion time: main.py, requirements.txt, Dockerfile, README.md,
  metadata.json, tools/ and tests/ (bytecode caches excluded).
- `logs/` holds the regression of this exact main.py: T1 (Mac and x86-64), T3 official protocol (log and per-mission
  JSON), T4 on the Phase 4 robustness inputs (log, JSON, verdicts), T4 on the held-out robustness set, T5 measurements
  (Apple M3 Pro and 2.1 GHz Xeon), T6 (container, fresh macOS environments, x86-64), T7 and the renderer test.
- `evidence/` holds the programme rules (README), the registry of every candidate and analysis (registry.json,
  REGISTRY.md), and the analysis outputs behind the promotion (paired official-protocol comparison, per-sequence table,
  robustness comparisons).
- `SHA256SUMS` lists every file in this folder except itself.

## Identity

- main.py SHA-256: c401d784ddc78e91ea6f76f394d5a93588c1d05e95c20a5482ab01cecac944e2 (97,476 bytes), `__version__ = "1.1.0"`
- Configuration: V5P4 + rest detector 50 % / 0.06 rad/s, no rotor-drag rows at rest, steady drag-offset prior
  (0.03 m/s^2, 1000 s, initial 0.15) when the loose rotor radius is >= 1.7 m and the radius guard (checked on the loose
  solution) passes, kept only if it passes the guard too; 24 s final hand-held gravity when the gyro is clean.

## Verified behaviour (Phase 5)

- Official protocol T3 (seeds 42 + 5-13, both orders, 340 missions): EuRoC ATE 2.294 m (v1.0.0 2.670, -14.1 %
  [-21.2, -4.8]), waypoints -15.8 %, after-dropout -13.3 %, blind -3.2 % (n.s.); TUM-VI ATE 0.418 m (0.451, -7.3 %
  [-13.8, +1.5]), waypoints -11.4 %; 0 contract failures or fallbacks.
- Promotion set CONF (seeds 3001-3010): pooled composite -11.21 % [-17.41, -3.52]; STRESS -9.10 % [-13.93, -3.15].
- T1 34/34 (macOS arm64 and Linux x86-64, byte-identical across machines); T2 986/986 byte-identical to the research build.
- T4: 935 + 799 held-out cases, 0 errors, 0 fallbacks, every family's mean ATE better than v1.0.0; one intended
  bias-mode change; the cases > 25 % worse are listed in the README limitations.
- T5: own heap 2.64 MB (3 min) / 6.86 MB (10 min); worst runtime 11.7 ms per IMU-second on the Xeon, 6.7 on the M3 Pro.
- T6: python:3.12-slim container, three fresh macOS environments (Python 3.10/3.13, NumPy 1.22.4/1.26.4/2.5.5) and
  Linux x86-64 (Python 3.13.18, NumPy 2.5.3): every output byte-identical. T7 37/37. Renderer 10^7 rows, 0 mismatches.

## Restore procedure

Copy `release/` from this folder (or only `release/main.py`) and check main.py against the SHA-256 above. To fall back to
v1.0.0, use `champion/v1.0.0/release/` (main.py SHA-256 f0d9e3c0...c7c6). Nothing has been published, pushed or submitted.
