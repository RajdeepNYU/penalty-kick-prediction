import numpy as np, pandas as pd, sys, pickle
sys.argv = ["x"]
from cv import X, Xm, gt, C
yk, yg, yo = gt["kick_direction"].values, gt["goalkeeper_direction"].values, gt["goal_outcome"].values
Pk = np.load(f"{C}/oof_final_kick.npy"); cl = np.array(["left", "middle", "right"])
pk = cl[Pk.argmax(1)]
Pg = np.load(f"{C}/oof_final_gk_gbdt.npy"); cnn = np.load(f"{C}/oof_gk_cnn30.npy")[:, [0, 2, 1]]
pg = np.array(["left", "right", "stands in the middle"])[(0.5 * Pg + 0.5 * cnn).argmax(1)]
print("=== 1. what would 0.80 require? ===")
for g, o in ((0.87, 0.722), (0.95, 0.722), (0.95, 0.85)):
    print(f"  keeper {g:.2f}, outcome {o:.3f} -> kick needed {(0.80 - 0.25 * g - 0.15 * o) / 0.6:.3f}")
mid = (yk == "middle").mean(); lr = yk != "middle"
print(f"  'middle' kicks are {mid:.1%} of data. If middle is never predicted, kick <= {1 - mid:.3f} even with perfect left/right.")
for target in (0.70, 0.75, 0.80): print(f"  kick {target:.2f} with 0 middle recall needs left/right-only accuracy {target / (1 - mid):.3f}")
print(f"  we have: left/right-only accuracy {(pk[lr] == yk[lr]).mean():.3f}, overall kick {(pk == yk).mean():.3f}")

print("\n=== 2. label-convention consistency per source (saves: keeper usually dives the shot side) ===")
d = gt.assign(yk=yk, yg=yg)
sv = d[d.outcome_detail == "save"]
print(f"  overall: among saves kick==keeper side {np.mean(sv.yk == sv.yg):.3f} (n={len(sv)}); among goals {np.mean(d[d.outcome_detail == 'goal'].yk == d[d.outcome_detail == 'goal'].yg):.3f}")
s = sv.groupby("source_id").agg(n=("yk", "size"), agree=("yk", lambda x: np.mean(x.values == sv.loc[x.index, 'yg'].values)))
print("  sources with >=4 saves:", int((s.n >= 4).sum()), "| agreement<0.4 in:", int(((s.n >= 4) & (s.agree < 0.4)).sum()), "sources ->", s[(s.n >= 4) & (s.agree < 0.4)].round(2).to_dict("index"))

print("\n=== 3. is per-source accuracy dispersion larger than chance? (label/camera inconsistency by source) ===")
d["ok"] = pk == yk
g = d.groupby("source_id").agg(n=("ok", "size"), acc=("ok", "mean")); g = g[g.n >= 8]
p = d.ok.mean(); exp_sd = np.sqrt(p * (1 - p) / g.n)
z = (g.acc - p) / exp_sd
print(f"  sources n>=8: {len(g)}; sd of per-source acc {g.acc.std():.3f} vs expected from binomial {np.sqrt((exp_sd ** 2).mean()):.3f}; share with |z|>2: {(z.abs() > 2).mean():.2f} (chance ~0.05)")
print("  worst sources:", g.sort_values("acc").head(6).round(2).to_dict("index"))
print("  best sources:", g.sort_values("acc").tail(6).round(2).to_dict("index"))

print("\n=== 4. is kick label skewed within source? (would be exploitable only with cross-clip info, which rules forbid) ===")
t = pd.crosstab(d.source_id, d.yk); t = t[t.sum(1) >= 8]
purity = (t.max(1) / t.sum(1)); base = (d.yk.value_counts(normalize=True) ** 2).sum()
print(f"  mean within-source majority-class share {purity.mean():.3f} (chance for 3 classes at these n ~{(t.sum(1).map(lambda n: 0)).mean() + 0.52:.2f}); share of sources where one class >=70%: {(purity >= .7).mean():.2f}")

print("\n=== 5. kick accuracy vs kick-phase proxy (ankle gap & leg spread in last frame) ===")
gap_u = np.abs(X[:, 90 + 8 + 8]) if False else None
ank = np.abs(X[:, 138]) if X.shape[1] > 138 else None   # |ankle-ankle forward gap| at last frame (lines block, frame 0)
q = pd.qcut(ank, 4, duplicates="drop")
print(d.assign(q=q).groupby("q", observed=True).agg(n=("ok", "size"), acc=("ok", "mean")).round(3).to_string())

print("\n=== 6. oracle diagnostics: how much of kick is explainable by the keeper? ===")
print("  true keeper side -> kick side agreement among left/right shots:", round(float(np.mean(yk[(yk != 'middle') & (yg != 'stands in the middle')] == yg[(yk != 'middle') & (yg != 'stands in the middle')])), 3))
print("  predicting kick = true keeper side would score", round(float(np.mean(np.where(yg == 'stands in the middle', 'left', yg) == yk)), 3), "(an oracle on keeper; ours is 0.87 accurate)")
