# Research code (Phases 1–4)

The scripts and result summaries behind `docs/PHASE3_FINAL_RESEARCH_AND_ARCHITECTURE_FREEZE.md` (Appendix A there maps every
official-data experiment (E40–E65) to its script and result file; earlier experiments follow the same naming) and `docs/PHASE4_BUILD_AND_VERIFICATION_REPORT.md`.

| Folder | Contents |
|---|---|
| `harness/` | research reference implementation (`fbs6.py`: front end, smoothers, estimator; `blind5.py`: blind-window rules) and shared utilities |
| `experiments/` | one script per experiment (`eNN_*.py`) |
| `results/` | result summaries (JSON) of those experiments |
| `phase4/` | the Phase 4 test scripts (T1–T7) and their result summaries |
| `poc/` | proofs of concept (memory governor, entry point, blind-row convention) |
| `acquire/` | scripts that download the public EuRoC MAV and TUM-VI sequences from their official sources |

Not included: dataset files and anything derived from them row by row (inputs, ground-truth extracts, caches, trajectories),
model weights, result files larger than 1 MB, and the organisers' `degrade_imu.py` (part of the participant package); scripts that
need it expect your own copy. Paths inside result files point to the team's workspace (`~/SamsungPRISM_FiveBlindSeconds/...`).
The submission code itself is `src/main.py`, which does not depend on anything in this folder.
