"""Deck figures for Thapar_FiveBlindSeconds (Five Nines palette, transparent background).

Inputs: official-format inputs data/phase4/inputs/*_A42.zip, v1.1.0 outputs opt/regression/t1_rcc/*.txt,
v1.0.0 outputs champion/v1.0.0/out_t2_reference_outputs.tar.xz, public EuRoC ground truth (scoring/plotting only).
"""
import io, json, os, sys, tarfile, zipfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = os.path.expanduser("~/SamsungPRISM_FiveBlindSeconds")
OUT = os.path.join(ROOT, "deck", "fig"); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "rc", "tools"))
import evaluate as ev  # noqa: E402

for f in os.listdir(os.path.join(ROOT, "deck", "fonts")):
    if f.endswith(".ttf"): fm.fontManager.addfont(os.path.join(ROOT, "deck", "fonts", f))
BG, LT1, DK2, LT2 = "#0E0D0B", "#EFE8DA", "#1D1B17", "#C9C1B2"
OR, MINT, AMB, BLUE, RUST, GREY = "#FF5A1F", "#6FD3B0", "#E3B04B", "#9CC3D9", "#C8643B", "#8D877C"
plt.rcParams.update({"font.family": "IBM Plex Sans", "font.size": 11, "text.color": LT2, "axes.labelcolor": LT2,
                     "xtick.color": GREY, "ytick.color": GREY, "axes.edgecolor": "#3A3630", "axes.facecolor": "none",
                     "figure.facecolor": "none", "savefig.transparent": True, "axes.linewidth": 0.8})

def est_v10(name):
    with tarfile.open(os.path.join(ROOT, "champion/v1.0.0/out_t2_reference_outputs.tar.xz")) as t:
        data = t.extractfile("out_t2/%s.txt" % name).read().decode()
    tmp = os.path.join(OUT, "_tmp_v10.txt"); open(tmp, "w").write(data); r = ev.read_estimate(tmp); os.remove(tmp); return r

def aligned(m, p):
    A = m.ok & np.isfinite(p).all(1); Rm, tt = ev.umeyama(p[A], m.gp[A]); return p @ Rm.T + tt

# ------------------------------------------------------------------ 1. trajectories: hardest drone flight + typical hand-held room
def _panel(ax, name, gtpath, title):
    inp = os.path.join(ROOT, "data/phase4/inputs/%s.zip" % name)
    tno = ev.read_input_ns(inp); m = ev.mission(tno, ev.read_gt(gtpath))
    n11, p11, q11 = ev.read_estimate(os.path.join(ROOT, "opt/regression/t1_rcc/%s.txt" % name))
    n10, p10, q10 = est_v10(name); assert (n11 == tno).all() and (n10 == tno).all()
    s11, s10 = ev.score(m, p11, q11), ev.score(m, p10, q10)
    s0 = ev.score(m, np.zeros_like(p11), np.tile([0, 0, 0, 1.0], (len(p11), 1)))
    a11, a10 = aligned(m, p11), aligned(m, p10); ok = m.ok; pre = ok & ~m.blind; bl = ok & m.blind
    Tp = m.T - 5.0; ci = [int(np.argmin(np.abs(np.where(pre, m.t, 1e9) - Tp * (k + 1) / 9))) for k in range(8)]
    ax.plot(m.gp[ok, 0], m.gp[ok, 1], color=BLUE, lw=2.0, alpha=0.8, label="Ground truth (used only to score)", zorder=2)
    ax.plot(a10[pre, 0], a10[pre, 1], color=GREY, lw=0.9, alpha=0.9, ls=(0, (3, 2)), label="v1.0.0  ATE %.2f m" % s10["ate"], zorder=3)
    ax.plot(a11[pre, 0], a11[pre, 1], color=OR, lw=1.5, label="v1.1.0  ATE %.2f m" % s11["ate"], zorder=4)
    ax.plot(m.gp[bl, 0], m.gp[bl, 1], color=LT1, lw=3.0, alpha=0.6, solid_capstyle="round", zorder=5)
    ax.plot(a11[bl, 0], a11[bl, 1], color=AMB, lw=3.2, solid_capstyle="round", label="5 s blind prediction", zorder=6)
    L = int(np.where(pre)[0][-1]); ax.scatter([a11[L, 0]], [a11[L, 1]], s=46, facecolor=BG, edgecolor=AMB, lw=1.8, zorder=7)
    ax.scatter(m.gp[ci, 0], m.gp[ci, 1], s=44, facecolor=BG, edgecolor=LT1, lw=1.1, zorder=8, label="8 scored waypoints")
    ax.set_aspect("equal"); ax.grid(color="#2A2722", lw=0.6); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    for sp in ax.spines.values(): sp.set_visible(False)
    leg = ax.legend(loc="best", frameon=False, fontsize=8.2, handlelength=2.2)
    for t in leg.get_texts(): t.set_color(LT2)
    ax.set_title(title + "\nmotionless floor %.2f m" % s0["ate"], color=LT1, fontsize=10.5, loc="left")
    return dict(name=name, ate_v11=s11["ate"], ate_v10=s10["ate"], ate_motionless=s0["ate"], wp_v11=s11["wp"], wp_motionless=s0["wp"],
                blind_v11=s11["b_abs"], blind_motionless=s0["b_abs"], ori_v11=s11["ori"], seconds=float(m.T), rows=int(len(tno)), blind_rows=int(m.blind.sum()))

def fig_trajectory():
    out = {}
    for name, gtp, title, fn, size in [
        ("MH_04_difficult_A42", "data/euroc/MH_04_difficult/mav0/state_groundtruth_estimate0/data.csv", "Drone \u00b7 EuRoC MH_04 difficult (19 m hall)", "trajectory_MH04.png", (6.2, 5.4)),
        ("V1_02_medium_A42", "data/euroc/V1_02_medium/mav0/state_groundtruth_estimate0/data.csv", "Drone \u00b7 EuRoC V1_02 medium (Vicon room)", "trajectory_V102.png", (6.2, 5.4)),
        ("room5_A42", "data/tumvi/dataset-room5_512_16/mav0/mocap0/data.csv", "Hand-held \u00b7 TUM-VI room5", "trajectory_room5.png", (6.2, 5.4))]:
        fig, ax = plt.subplots(figsize=size, dpi=220)
        out[name] = _panel(ax, name, os.path.join(ROOT, gtp), title)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, fn)); plt.close(fig)
    return out

# ------------------------------------------------------------------ 2. hero path (decorative, same flight, no axes)
def fig_hero(name="MH_04_difficult_A42", seq="MH_04_difficult"):
    inp = os.path.join(ROOT, "data/phase4/inputs/%s.zip" % name)
    gt = ev.read_gt(os.path.join(ROOT, "data/euroc/%s/mav0/state_groundtruth_estimate0/data.csv" % seq))
    tno = ev.read_input_ns(inp); m = ev.mission(tno, gt)
    _n, p11, _q = ev.read_estimate(os.path.join(ROOT, "opt/regression/t1_rcc/%s.txt" % name))
    a11 = aligned(m, p11); ok = m.ok; pre = ok & ~m.blind; bl = ok & m.blind
    fig, ax = plt.subplots(figsize=(7.0, 7.0), dpi=220)
    ax.plot(m.gp[ok, 0], m.gp[ok, 1], color=LT2, lw=0.8, alpha=0.35)
    ax.plot(a11[pre, 0], a11[pre, 1], color=OR, lw=1.3, alpha=0.95)
    ax.plot(a11[bl, 0], a11[bl, 1], color=AMB, lw=6.0, solid_capstyle="round", zorder=4)
    L = int(np.where(pre)[0][-1]); ax.scatter([a11[L, 0]], [a11[L, 1]], s=150, facecolor=BG, edgecolor=AMB, lw=2.6, zorder=5)
    ax.scatter([a11[-1, 0]], [a11[-1, 1]], s=90, facecolor=AMB, edgecolor=AMB, lw=0, zorder=6)
    ax.set_aspect("equal"); ax.axis("off"); fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(os.path.join(OUT, "hero_path.png")); plt.close(fig)

# ------------------------------------------------------------------ 3. what the IMU looks like (official input)
def fig_signal(name="MH_01_easy_A42"):
    z = zipfile.ZipFile(os.path.join(ROOT, "data/phase4/inputs/%s.zip" % name))
    rows = [ln for ln in io.TextIOWrapper(z.open("imu0/data.csv"), encoding="utf-8") if ln and ln[0].isdigit()]
    d = np.array([[float(x) for x in ln.split(",")] for ln in rows]); ns = np.array([int(ln.split(",", 1)[0]) for ln in rows], np.int64)
    t = (ns - ns[0]) * 1e-9; a = d[:, 4:7]; w = d[:, 1:4]
    if abs(np.median(np.linalg.norm(w, axis=1)) - 9.8) < abs(np.median(np.linalg.norm(a, axis=1)) - 9.8): a, w = w, a
    an = np.linalg.norm(a, axis=1); zero = np.all(d[:, 1:7] == 0, axis=1)
    dt = np.diff(ns) * 1e-9; med = np.median(dt[dt > 0]); miss = np.round(dt / med).astype(int) - 1
    gaps = [(t[i], dt[i]) for i in np.where(miss >= 1)[0]]
    big = [(t0, g) for t0, g in gaps if g >= 0.5]
    T = t[-1]; blind0 = t[np.where(~zero)[0][-1]] if zero.any() else T
    fig, ax = plt.subplots(figsize=(12.0, 3.0), dpi=200)
    v = ~zero
    ax.plot(t[v], an[v], color=LT2, lw=0.35, alpha=0.75, rasterized=True)
    for t0, g in big: ax.axvspan(t0, t0 + g, color=RUST, alpha=0.55, lw=0)
    for t0, g in gaps:
        if g < 0.5: ax.plot([t0, t0], [0.4, 1.6], color=RUST, lw=0.5, alpha=0.6)
    ax.axvspan(blind0, T, color=AMB, alpha=0.30, lw=0)
    k = 21; hp = an - np.convolve(an, np.ones(k) / k, mode="same"); W = 400; idx = np.where(v)[0]
    cs = [(t[idx[j]], np.std(hp[idx[j:j + W]])) for j in range(0, len(idx) - W, W // 2)]
    ax2 = ax.twinx(); ax2.plot([c[0] for c in cs], [c[1] for c in cs], color=OR, lw=1.6)
    ax2.set_ylabel("noise sd (m/s²)", color=OR); ax2.tick_params(colors=OR); ax2.set_ylim(0, max(c[1] for c in cs) * 1.15)
    for sp in ["top", "left", "bottom"]: ax2.spines[sp].set_visible(False)
    ax2.spines["right"].set_color("#3A3630"); noise_first = float(np.mean([c[1] for c in cs[:6]])); noise_last = float(np.mean([c[1] for c in cs[-6:]]))
    ax.set_xlim(0, T + 0.5); lo, hi = np.percentile(an[v], [0.1, 99.9]); ax.set_ylim(0, max(hi * 1.08, 20))
    ax.set_xlabel("time (s)"); ax.set_ylabel("|a| (m/s²)"); ax.grid(axis="y", color="#2A2722", lw=0.6)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "signal_MH01.png")); plt.close(fig)
    nz_first = an[v][t[v] < 20]; nz_last = an[v][(t[v] > blind0 - 25) & (t[v] < blind0)]
    return dict(name=name, seconds=float(T), rows=int(len(ns)), gaps=len(gaps), gaps_ge_5=int(sum(1 for i in np.where(miss >= 5)[0])),
                long_gaps=[(round(t0, 2), round(g, 2)) for t0, g in big], blind_start=float(blind0), blind_len=float(T - blind0),
                zero_rows=int(zero.sum()), noise_first=noise_first, noise_last=noise_last, sd_first20=float(np.std(nz_first)), sd_last25=float(np.std(nz_last)), med_dt_ms=float(med * 1e3))

# ------------------------------------------------------------------ 4. what the jammer adds: degraded minus clean public IMU (same rows)
def fig_injected(name="MH_01_easy_A42", seq="MH_01_easy"):
    z = zipfile.ZipFile(os.path.join(ROOT, "data/phase4/inputs/%s.zip" % name))
    rows = [ln for ln in io.TextIOWrapper(z.open("imu0/data.csv"), encoding="utf-8") if ln and ln[0].isdigit()]
    D = np.array([[float(x) for x in ln.split(",")[1:]] for ln in rows]); N = np.array([int(ln.split(",", 1)[0]) for ln in rows], np.int64)
    C = [ln for ln in open(os.path.join(ROOT, "data/euroc/%s/mav0/imu0/data.csv" % seq)) if ln[0].isdigit()]
    CD = np.array([[float(x) for x in ln.split(",")[1:]] for ln in C]); CN = np.array([int(ln.split(",", 1)[0]) for ln in C], np.int64)
    i = np.clip(np.searchsorted(CN, N), 1, len(CN) - 1); j = np.where(np.abs(CN[i - 1] - N) < np.abs(CN[i] - N), i - 1, i)
    t = (N - N[0]) * 1e-9; zero = np.all(D == 0, 1); v = ~zero; e = D[:, 3:6] - CD[j, 3:6]
    T = t[-1]; blind0 = t[np.where(v)[0][-1]]
    dtn = np.diff(N) * 1e-9; med = np.median(dtn[dtn > 0]); miss = np.round(dtn / med).astype(int) - 1; gi = np.where(miss >= 1)[0]
    plt.rcParams.update({"font.size": 8})
    fig, ax = plt.subplots(figsize=(7.0, 1.75), dpi=300)
    ax.plot(t[v], e[v, 0], color=LT2, lw=0.25, alpha=0.75, rasterized=True)
    W = 400; idx = np.where(v)[0]; cm = [(t[idx[k + W // 2]], np.mean(e[idx[k:k + W], 0]), np.std(e[idx[k:k + W], 0])) for k in range(0, len(idx) - W, W // 4)]
    ax.plot([c[0] for c in cm], [c[1] for c in cm], color=OR, lw=1.8)
    ax.axvspan(blind0, T, color=AMB, alpha=0.55, lw=0)
    ylo, yhi = np.percentile(e[v, 0], [0.2, 99.8]); pad = 0.15 * (yhi - ylo)
    for k in gi: ax.plot([t[k], t[k]], [ylo - pad * 0.9, ylo - pad * 0.3], color=RUST, lw=0.5, alpha=0.7)
    ax.set_ylim(ylo - pad, yhi + pad * 0.5); ax.set_xlim(0, T + 0.5)
    ax.set_xlabel("time (s)"); ax.set_ylabel("added error (m/s²)"); ax.grid(axis="y", color="#2A2722", lw=0.5)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.3); fig.savefig(os.path.join(OUT, "injected_MH01.png")); plt.close(fig); plt.rcParams.update({"font.size": 11})
    q = len(cm) // 10
    return dict(sd_first=float(np.mean([c[2] for c in cm[:q]])), sd_last=float(np.mean([c[2] for c in cm[-q:]])),
                bias_first=float(np.mean([c[1] for c in cm[:q]])), bias_last=float(np.mean([c[1] for c in cm[-q:]])), gaps=int(len(gi)),
                missing_rows=int(miss[gi].sum()), blind_s=float(T - blind0))

if __name__ == "__main__":
    info = dict(trajectory=fig_trajectory(), signal=fig_signal(), injected=fig_injected()); fig_hero()
    json.dump(info, open(os.path.join(OUT, "figures.json"), "w"), indent=1); print(json.dumps(info, indent=1))
