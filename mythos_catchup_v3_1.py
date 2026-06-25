"""
================================================================================
Mythos catch-up model · v3.1 (canonical, consolidated)
================================================================================
Change from v3.0: hyperscaler flagship budget gamma_A upper bound raised
0.015 -> 0.020. Rationale: the revealed reference class tops out at DeepSeek
~1.6%, which the old 1.5% ceiling excluded; 2.0% covers the observed maximum
plus headroom for disclosure noise, while keeping the mode at the BAU 1.0%.
Serious behavioural outliers (seed-push, crash, xAI-style) are handled in the
text and in the regime ladder (fig 4), NOT integrated into the baseline.

Single-baseline framing: ONE forecast of when a Chinese firm has a Mythos-like
model ready for use, from a race between two archetypes sharing the same
algorithmic-progress and target draws:

  TRACK A - top hyperscaler: aggregate pool = domestic stock + flows + remote
            access; flagship budget gamma_A ~ Tri(0.5%, 1.0%, 2.0%) of annual
            throughput (revealed: Meta 0.5-0.8%, Qwen 0.2-0.6%, DeepSeek ~1-1.6%)
  TRACK B - top pure-play lab (DeepSeek-shaped): stock ~50-110k H100e growing;
            gamma_B ~ Tri(3%, 6%, 12%) (revealed-to-spend construct span)

Reduced form: a run launches at the first t where
      mythos_flops / alg^(t/12)  <=  12 * gamma * pool(t) * F_EFF
then takes max(compute time, T_min) months, plus T_post for post-training.
Forecast = min over tracks, + T_post.  t=0 = 24 Feb 2026 (Mythos internally
available).  Policy counterfactuals are expressed as LEVERS on this baseline.
================================================================================
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date, timedelta

# ----------------------------------------------------------------------------
# constants & helpers
# ----------------------------------------------------------------------------
N          = 100_000
H100_PEAK  = 9.89e14
SEC_MONTH  = 30.44 * 24 * 3600
T_GRID     = np.arange(0, 60.5, 0.5)
T0         = date(2026, 2, 24)
TODAY      = date(2026, 6, 12)
TODAY_M    = (TODAY - T0).days / 30.44
OUT        = "/mnt/user-data/outputs"
import os; os.makedirs(OUT, exist_ok=True)

def m2date(m): return T0 + timedelta(days=float(m) * 30.44)
def m2s(m):    return m2date(m).strftime("%b %d, %Y")

def tri_ppf(u, a, c, b):
    Fc = (c - a) / (b - a)
    return np.where(u < Fc, a + np.sqrt(u * (b - a) * (c - a)),
                            b - np.sqrt((1 - u) * (b - a) * (b - c)))

# ----------------------------------------------------------------------------
# parameters: (min, mode, max) triangulars; alg is triangular in log10
# ----------------------------------------------------------------------------
P = {
 "duration_anthropic": (2.5, 3.0, 3.5),
 "mfu_anthropic":      (0.15, 0.20, 0.30),
 "mfu_china":          (0.15, 0.20, 0.30),
 "gamma_A":            (0.005, 0.010, 0.020), # v3.1: upper 1.5%->2.0% (covers DeepSeek ~1.6% + headroom)
 "total_in_china":     (1.4e6, 1.78e6, 2.7e6),
 "top_firm_share":     (0.20, 0.27, 0.35),
 "h200":               (70e3, 75e3, 75e3),
 "D_h200":             (3.0, 6.0, 12.0),
 "PR26":               (280e3, 345e3, 415e3),
 "D_pr":               (6.0, 12.0, 24.0),
 "top_pr_share":       (0.20, 0.30, 0.45),
 "smug26":             (190e3, 480e3, 1300e3),
 "remote_t0":          (0.8e6, 1.0e6, 1.2e6),
 "remote_growth":      (0.3e6, 0.5e6, 0.8e6),
 "D_rem":              (9.0, 12.0, 15.0),
 "remote_concen":      (0.40, 0.50, 0.60),
 "gamma_B":            (0.03, 0.06, 0.12),
 "lab0":               (50e3, 75e3, 110e3),
 "lab_growth":         (30e3, 75e3, 150e3),
 "D_lab":              (9.0, 12.0, 18.0),
 "T_min":              (0.75, 1.5, 3.0),
 "T_post":             (0.5, 1.5, 3.0),
 "DT26":               (150e3, 184e3, 215e3),
 "PR27":               (370e3, 460e3, 540e3),
 "rubin":              (1.2, 1.5, 1.9),
}
ALG = (np.log10(2), np.log10(10), np.log10(50))

def draw_all(n, rng):
    v = {k: tri_ppf(rng.random(n), *spec) for k, spec in P.items()}
    v["alg"] = 10 ** tri_ppf(rng.random(n), *ALG)
    return v

def medians():
    v = {k: np.array([tri_ppf(np.array([0.5]), *spec)[0]]) for k, spec in P.items()}
    v["alg"] = 10 ** tri_ppf(np.array([0.5]), *ALG)
    return v

# ----------------------------------------------------------------------------
# engine
# ----------------------------------------------------------------------------
def simulate(v, remote="full", smuggle_stop=False, h200_stop=False,
             ascend27_off=False, distill_r=None, keep_pools=False, t_grid=None):
    n = len(v["alg"])
    tg = T_GRID if t_grid is None else t_grid
    T = tg[None, :]
    alg = v["alg"] if distill_r is None else v["alg"] ** (1.0 - distill_r)
    N_AH = 500_000 * (6.5e14 / 9.89e14)
    mflops = N_AH * H100_PEAK * v["mfu_anthropic"] * v["duration_anthropic"] * SEC_MONTH
    F_EFF  = H100_PEAK * v["mfu_china"] * SEC_MONTH
    myt = mflops[:, None] / alg[:, None] ** (T / 12.0)

    cap = lambda t: np.minimum(t, TODAY_M)
    ramp = lambda tot, s, d: tot[:, None] * np.clip((T - s) / d, 0.0, 1.0)

    h200_t = cap(T) if h200_stop else T
    smug_t = cap(T) if smuggle_stop else T
    dom = (v["total_in_china"] * v["top_firm_share"])[:, None] \
        + v["h200"][:, None] * np.minimum(h200_t / v["D_h200"][:, None], 1.0) \
        + (v["PR26"] * v["top_pr_share"])[:, None] * np.minimum(T / v["D_pr"][:, None], 1.0) \
        + (v["smug26"] * v["top_firm_share"])[:, None] * np.clip(smug_t / 12.0, 0, 1)
    if not smuggle_stop:
        dom = dom + ramp(v["smug26"] * v["rubin"] * v["top_firm_share"], 12.0, 12.0)
    if not ascend27_off:
        dom = dom + ramp(v["DT26"] * v["top_pr_share"], 7.0, 12.0) \
                  + ramp(v["PR27"] * v["top_pr_share"], 10.0, 12.0)

    rem_t = {"full": T, "frozen": cap(T)}.get(remote)
    if remote == "revoked":
        rem = np.zeros_like(dom)
    else:
        rem = (v["remote_t0"] * v["remote_concen"])[:, None] \
            + (v["remote_growth"] * v["remote_concen"] / v["D_rem"])[:, None] * np.minimum(rem_t, 24.0)
    lab = v["lab0"][:, None] + (v["lab_growth"] / v["D_lab"])[:, None] * np.minimum(T, 24.0)

    aN = np.arange(n)
    def track(pool, g):
        T_run = myt / (pool * F_EFF[:, None])
        tot = np.where(T_run <= 12.0 * g[:, None],
                       T + np.maximum(T_run, v["T_min"][:, None]), np.inf)
        i = tot.argmin(axis=1)
        return tot[aN, i], tg[i], myt[aN, i]
    tA, tA_start, tA_flops = track(dom + rem, v["gamma_A"])
    tB, _, _               = track(lab,       v["gamma_B"])
    pre = np.minimum(tA, tB)
    out = {"pre": pre, "ready": pre + v["T_post"], "winner_A": tA < tB,
           "tA": tA, "tB": tB, "A_start": tA_start, "A_flops": tA_flops,
           "mflops": mflops, "F_EFF": F_EFF, "alg_eff": alg}
    if keep_pools:
        out["pools"] = (dom, rem, lab)
    return out

# ----------------------------------------------------------------------------
# RUN: baseline + levers
# ----------------------------------------------------------------------------
rng = np.random.default_rng(42)
V = draw_all(N, rng)
u_dist = rng.random(N)
distill_r = tri_ppf(rng.random(N), 0.10, 0.25, 0.40) * tri_ppf(rng.random(N), 0.30, 0.50, 0.70)

base = simulate(V, keep_pools=True)
def lever_ready(**kw):
    return simulate(V, **kw)["ready"]
levers = [
 ("Remote access revoked",        lever_ready(remote="revoked")),
 ("Distillation limited",         lever_ready(distill_r=distill_r)),
 ("Remote revoked + distillation",lever_ready(remote="revoked", distill_r=distill_r)),
 ("Remote growth frozen",         lever_ready(remote="frozen")),
 ("Smuggling flows stopped",      lever_ready(smuggle_stop=True)),
 ("H200 shipments stopped",       lever_ready(h200_stop=True)),
 ("Ascend 2027 cohorts removed",  lever_ready(ascend27_off=True)),
]

print("=" * 96)
print(f"Mythos catch-up model v3.1 · N = {N:,} · t=0 = {T0} · today = t+{TODAY_M:.1f} mo")
print("=" * 96)
print("\nBASELINE (world as it is):")
for ev in ["pre", "ready"]:
    q = np.percentile(base[ev], [5, 25, 50, 75, 95])
    print(f"  {ev:>5}: " + "  ".join(f"P{p} {x:5.1f} ({m2s(x)})" for p, x in zip([5,25,50,75,95], q)))
print(f"  hyperscaler wins race: {base['winner_A'].mean():.0%}   P(<today): {(base['ready']<TODAY_M).mean():.1%}")
for lbl, thr in [("Nov 2026", (date(2026,11,30)-T0).days/30.44), ("end-2026", (date(2026,12,31)-T0).days/30.44)]:
    print(f"  P(ready <= {lbl}): {(base['ready']<thr).mean():.1%}")

print("\nLEVER LADDER (delta on 'ready', months):")
b = np.percentile(base["ready"], [50, 75, 95])
for nm, r in levers:
    q = np.percentile(r, [50, 75, 95])
    print(f"  {nm:<32} P50 {q[0]:5.1f} ({m2s(q[0])})   d: {q[0]-b[0]:+.1f} / {q[1]-b[1]:+.1f} / {q[2]-b[2]:+.1f}")

# ----------------------------------------------------------------------------
# TORNADO
# ----------------------------------------------------------------------------
TORNADO = [
 ("Algorithmic progress",       "alg"),
 ("Behavioural budget γ_A",     "gamma_A"),
 ("Lab budget γ_B",             "gamma_B"),
 ("MFU ratio (China/Anthropic)","__ratio__"),
 ("Mythos training duration",   "duration_anthropic"),
 ("Total in-China stock",       "total_in_china"),
 ("Top-firm share",             "top_firm_share"),
 ("Remote access at t=0",       "remote_t0"),
 ("Remote concentration",       "remote_concen"),
 ("Remote growth",              "remote_growth"),
 ("Smuggling 2026",             "smug26"),
 ("Lab stock at t=0",           "lab0"),
 ("Lab growth",                 "lab_growth"),
 ("Huawei 950PR 2026",          "PR26"),
 ("Min run duration",           "T_min"),
 ("Post-training time",         "T_post"),
]
FINE_GRID = np.arange(0, 60.001, 0.05)
def det_ready(overrides):
    v = medians()
    for k, val in overrides.items():
        v[k] = np.array([val])
    return simulate(v, t_grid=FINE_GRID)["ready"][0]

base_det = det_ready({})
rows = []
for label, key in TORNADO:
    if key == "__ratio__":
        ratio = V["mfu_anthropic"] / V["mfu_china"]
        r10, r90 = np.percentile(ratio, [10, 90])
        mc = medians()["mfu_china"][0]
        lo = det_ready({"mfu_anthropic": mc * r10}); hi = det_ready({"mfu_anthropic": mc * r90})
    elif key == "alg":
        a10, a90 = (10 ** tri_ppf(np.array([q]), *ALG)[0] for q in (0.1, 0.9))
        lo, hi = det_ready({"alg": a10}), det_ready({"alg": a90})
    else:
        p10, p90 = (tri_ppf(np.array([q]), *P[key])[0] for q in (0.1, 0.9))
        lo, hi = det_ready({key: p10}), det_ready({key: p90})
    rows.append((label, lo, hi))
rows.sort(key=lambda r: abs(r[2] - r[1]), reverse=True)

print(f"\nTORNADO (deterministic baseline ready = {base_det:.1f} mo; param at own P10 / P90):")
for label, lo, hi in rows:
    print(f"  {label:<28} {lo:5.1f}  <->  {hi:5.1f}   (range {abs(hi-lo):.1f})")

v = medians(); s = simulate(v, t_grid=FINE_GRID)
print(f"\nVERIFICATION (all inputs at medians): pre {s['pre'][0]:.1f}, ready {s['ready'][0]:.1f} "
      f"vs MC medians {np.median(base['pre']):.1f} / {np.median(base['ready']):.1f}")

# ============================================================================
# FIGURES
# ============================================================================
C_BASE, C_CF, C_LAB, C_REQ, C_GREY = "#8C2D04", "#5B6770", "#2A7F7F", "#6A3FB5", "#9AA0A6"
plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})

def cross_date(tg, req, bud):
    d = req - bud; sgn = np.where(np.diff(np.sign(d)))[0]
    if len(sgn) == 0: return None
    i = sgn[0]; f = d[i] / (d[i] - d[i + 1]); return tg[i] + f * (tg[i + 1] - tg[i])

# ---- Fig 1: headline CDF ----
fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=150)
grid = np.arange(0, 30.01, 0.25)
dts = [m2date(m) for m in grid]
cdf = [(base["ready"] < m).mean() * 100 for m in grid]
ax.plot(dts, cdf, color=C_BASE, lw=2.6)
for p, lbl in [(25, "P25"), (50, "median"), (75, "P75")]:
    m = np.percentile(base["ready"], p)
    ax.plot([m2date(m)], [p], "o", color=C_BASE, ms=6)
    ax.annotate(f"{lbl}: {m2date(m).strftime('%b %Y')}", (m2date(m), p),
                xytext=(10, -3), textcoords="offset points", fontsize=9, color=C_BASE)
ax.axvline(TODAY, color=C_GREY, lw=1, ls="--")
ax.text(TODAY + timedelta(days=8), 88, "today", color="#777", fontsize=9)
for off in (7, 8.2):
    ax.plot([m2date(off)], [1.5], marker="^", color="#B45309", ms=7, clip_on=False)
ax.annotate("naive trendline expectations\n(Epoch 7-mo · CAISI ~8-mo gaps)", (m2date(7.6), 3),
            xytext=(-10, 26), textcoords="offset points", fontsize=8, color="#B45309", ha="right")
ax.set_ylim(0, 102); ax.set_ylabel("Cumulative probability (%)")
ax.set_title("When will a Chinese firm have a Mythos-like model ready for use?", loc="left", fontsize=12)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.25)
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_headline_cdf.png"); plt.close()

# ---- Fig 2: lever ladder ----
fig, ax = plt.subplots(figsize=(8.4, 4.2), dpi=150)
b50 = np.percentile(base["ready"], 50)
order = [(nm, np.percentile(r, 50)) for nm, r in levers]
order.sort(key=lambda x: x[1] - b50)
colors = {"Remote access revoked": C_BASE, "Distillation limited": "#B45309",
          "Remote revoked + distillation": "#5B0E0E"}
for i, (nm, q50) in enumerate(order):
    c = colors.get(nm, C_GREY)
    ax.plot([m2date(b50), m2date(q50)], [i, i], color=c, lw=4, alpha=0.45, solid_capstyle="round")
    ax.plot([m2date(q50)], [i], "o", color=c, ms=8)
    ax.annotate(f"+{q50-b50:.1f} mo", (m2date(q50), i), xytext=(8, -3),
                textcoords="offset points", fontsize=9, color=c, fontweight="bold")
    ax.text(m2date(b50) - timedelta(days=10), i, nm, ha="right", va="center", fontsize=9)
ax.axvline(m2date(b50), color="#333", lw=1.2)
ax.text(m2date(b50), -0.85, f"baseline · {m2date(b50).strftime('%b %Y')}", ha="center", fontsize=8.5, color="#333")
ax.set_yticks([]); ax.set_xlim(m2date(8.0), m2date(16.5)); ax.set_ylim(-1.1, len(order) - 0.3)
ax.set_title("What each policy lever buys: shift in the median 'model ready' date\nLevers applied from today — only remote access has recoverable existing stock",
             loc="left", fontsize=11)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
ax.spines[["top", "right", "left"]].set_visible(False); ax.grid(axis="x", alpha=0.25)
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_lever_ladder.png"); plt.close()

# ---- Fig 3: gate / collision (crossings computed from data) ----
fig, ax = plt.subplots(figsize=(8.4, 5.0), dpi=150)
tg = np.arange(0, 24.01, 0.25)
gdts = [m2date(m) for m in tg]
idx = (tg / 0.5).astype(int)
req = base["mflops"][:, None] / V["alg"][:, None] ** (tg[None, :] / 12.0)
rp10, rp50, rp90 = np.percentile(req, [10, 50, 90], axis=0)
ax.plot(gdts, rp50, color=C_REQ, lw=2.3, label="Required FLOPs (Mythos-equivalent, decaying)")
ax.fill_between(gdts, rp10, rp90, color=C_REQ, alpha=0.12, lw=0)
dom, rem, lab = base["pools"]
for pool, g, c, ls, lbl, dx, dy, ha in [
    (dom + rem, V["gamma_A"], C_BASE, "-",  "Hyperscaler flagship budget (baseline, incl. remote)", 8, 12, "left"),
    (dom,       V["gamma_A"], C_CF,   "--", "Counterfactual: remote access revoked", 9, -20, "left"),
    (lab,       V["gamma_B"], C_LAB,  "-",  "Top lab flagship budget", 9, 15, "left")]:
    budp50 = np.percentile(12.0 * g[:, None] * pool[:, idx] * base["F_EFF"][:, None], 50, axis=0)
    ax.plot(gdts, budp50, color=c, lw=1.9, ls=ls, label=lbl)
    tc = cross_date(tg, rp50, budp50)
    if tc is not None:
        yv = np.interp(tc, tg, rp50)
        ax.plot([m2date(tc)], [yv], "o", color=c, ms=7, zorder=5)
        ax.annotate(m2date(tc).strftime("%b %Y"), (m2date(tc), yv), xytext=(dx, dy),
                    textcoords="offset points", color=c, fontsize=8.5, fontweight="bold", ha=ha)
ax.set_yscale("log"); ax.set_ylabel("FLOPs (log scale)"); ax.set_ylim(2e25, 1.5e27)
ax.set_title("The gate: a run launches when the decaying requirement falls below\na firm's flagship budget (γ × compute pool × annual throughput)", loc="left", fontsize=11.5)
ax.legend(frameon=False, fontsize=8.2, loc="upper right")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.2)
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_gate.png"); plt.close()

# ---- Fig 4: behavioural regime ladder (BAU row = new baseline band) ----
fig, ax = plt.subplots(figsize=(8.2, 3.8), dpi=150)
u_gA = rng.random(N)
regimes = [
 ("Observed BAU (γ ≈ 1%)\nMeta · Qwen · DeepSeek revealed", tri_ppf(u_gA, 0.005, 0.010, 0.020), "#37474F"),
 ("Seed-like push (γ ≈ 2%)",                                tri_ppf(u_gA, 0.010, 0.020, 0.030), "#B45309"),
 ("Crash programme (γ 3–8%)",                               tri_ppf(u_gA, 0.030, 0.050, 0.080), "#8C2D04"),
 ("xAI-style mobilisation (γ 10–27%)\nGrok 4 revealed",     tri_ppf(u_gA, 0.10, 0.16, 0.27),    "#5B0E0E"),
]
dom_full, rem_full, lab_full = base["pools"]
myt_b = base["mflops"][:, None] / V["alg"][:, None] ** (T_GRID[None, :] / 12.0)
aN = np.arange(N)
def track_g(g):
    T_run = myt_b / ((dom_full + rem_full) * base["F_EFF"][:, None])
    tot = np.where(T_run <= 12.0 * g[:, None], T_GRID[None, :] + np.maximum(T_run, V["T_min"][:, None]), np.inf)
    return tot[aN, tot.argmin(axis=1)]
for i, (lbl, g, c) in enumerate(regimes):
    ready = np.minimum(track_g(g), base["tB"]) + V["T_post"]
    q25, q50, q75 = np.percentile(ready, [25, 50, 75])
    y = len(regimes) - 1 - i
    ax.plot([m2date(q25), m2date(q75)], [y, y], color=c, lw=6, alpha=0.30, solid_capstyle="round")
    ax.plot([m2date(q50)], [y], "o", color=c, ms=9)
    ax.annotate(m2date(q50).strftime("%b %Y"), (m2date(q50), y), xytext=(0, 11),
                textcoords="offset points", ha="center", color=c, fontsize=9, fontweight="bold")
    ax.text(m2date(1.2), y, lbl, ha="right", va="center", fontsize=8.6)
ax.axvline(TODAY, color=C_GREY, lw=1, ls="--")
ax.text(TODAY, -0.75, "today", color="#777", fontsize=8.5, ha="center")
ax.set_yticks([]); ax.set_xlim(m2date(-7.5), m2date(19)); ax.set_ylim(-0.9, 3.7)
ax.set_title("The timeline is set by willingness, not chips:\nmedian 'model ready' date by commitment regime", loc="left", fontsize=11.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
ax.spines[["top", "right", "left"]].set_visible(False); ax.grid(axis="x", alpha=0.25)
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_regime_ladder.png"); plt.close()

# ---- Fig 5: tornado ----
fig, ax = plt.subplots(figsize=(8.4, 5.6), dpi=150)
show = rows[:12]
from datetime import timedelta as _td
for i, (label, lo, hi) in enumerate(show):
    y = len(show) - 1 - i
    dlo, dhi = m2date(lo), m2date(hi)
    ax.plot([dlo, dhi], [y, y], color="#D5D7DB", lw=5, solid_capstyle="round")
    ax.plot([dlo], [y], "o", color="#C2571A", ms=7)
    ax.plot([dhi], [y], "o", color="#2E7D52", ms=7)
    ax.text(min(dlo, dhi) - _td(days=10), y, label, ha="right", va="center", fontsize=8.8)
    ax.annotate(dlo.strftime("%b %y"), (dlo, y), xytext=(0, -13), textcoords="offset points", ha="center", fontsize=7.5, color="#C2571A")
    ax.annotate(dhi.strftime("%b %y"), (dhi, y), xytext=(0, 8),  textcoords="offset points", ha="center", fontsize=7.5, color="#2E7D52")
ax.axvline(m2date(base_det), color="#333", lw=1.2)
ax.text(m2date(base_det), -1.05, f"baseline · {m2date(base_det).strftime('%b %Y')}", ha="center", fontsize=8.5, color="#333")
ax.set_ylim(-1.3, len(show) - 0.4)
ax.set_yticks([]); ax.set_xlabel("Median 'model ready' date")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
ax.set_title("Uncertainty drivers: model outcome with each parameter pushed\nto its own 10th (orange) / 90th (green) percentile, all else at medians", loc="left", fontsize=11.5)
ax.spines[["top", "right", "left"]].set_visible(False); ax.grid(axis="x", alpha=0.25)
plt.tight_layout(); plt.savefig(f"{OUT}/fig5_tornado.png"); plt.close()

print("\nfigures written: fig1_headline_cdf, fig2_lever_ladder, fig3_gate, fig4_regime_ladder, fig5_tornado")
