"""Final assessment of the submission recipe N6 (kick = N4's: geometric blend of base and D pixel model; keeper = G3), read-only, no private labels exist or are used.
Out of fold on four partition sets (012, 345, 181920, 212223; source-grouped 5-fold, the RULES_nov6 judge's construction; the 12-clip camera correction is left out as there).
Part A, is the model guessing?  A1 exact tests against uniform guessing; A2 label permutation tests (all clips, and within each source); A3 paired tests and simulated
private boards against label-free rules (chance.csv's public score 0.52277 equals the PDF's "educated guess" baseline); A4 confidence vs accuracy; A5 per-group check;
A6 the public board. Part B, how close will private be to public?  B1 closed form for the gap; B2 simulated boards with the real test source sizes; B3 forecast of the
private score given the public one (method of ovf_y_intervals.py, 2 Nov, backtested 80.3% / 95.4%); B4 what the pick of the file did to the public score.
usage: python final_assess.py -> logs/final/assess.txt, cache/final/assess.pkl"""
import os, time, pickle, numpy as np, pandas as pd
from scipy.stats import norm, binomtest, spearmanr
from common import *
from cvlib import cl, yk
from ovf_lib import oth, Pk, cfg

os.makedirs(f"{P}/logs/final", exist_ok=True); os.makedirs(f"{PC}/final", exist_ok=True)
L = open(f"{P}/logs/final/assess.txt", "w", encoding="utf-8"); pr = lambda *a: (print(*a, flush=True), print(*a, file=L, flush=True))
rng = np.random.default_rng(20261106)

# ---------------- out-of-fold predictions of N6 on the four sets ----------------
SETS = ("012", "345", "181920", "212223"); SUF = {s: "" if s == "012" else f"_sd{s}" for s in SETS}; X2, NV = f"{PC}/px2", f"{PC}/nov6"
norm_ = lambda Q: Q / Q.sum(1, keepdims=True); geo = lambda x, y: norm_(np.exp(.5 * np.log(x + 1e-4) + .5 * np.log(y + 1e-4)))
tree1 = lambda S, s: (np.load(f"{PC}/oof_hgb_{S}_b2R{SUF[s]}.npy") + np.load(f"{PC}/oof_et_{S}_b2R{SUF[s]}.npy")) / 2
net = lambda sf: sum(np.load(f"{PC}/oofP_{e}_{h}_st{st}_{q}{sf}.npy") for e, h in cfg for st in ("EF", "G") for q in ("b2", "b2RM")) / 20
NN = {"012": net(""), "345": net("_sd345")}; NN["181920"] = NN["212223"] = (NN["012"] + NN["345"]) / 2
PXD = {"012": np.load(f"{PC}/oof_px_cat3D.npy"), "345": np.load(f"{PC}/oof_px_cat3D_sd345.npy"), "181920": np.load(f"{X2}/px_T_181920.npz")["D"], "212223": np.load(f"{X2}/px_T_212223.npz")["D"]}
QK = {s: geo(np.where(oth[:, None], Pk, .5 * sum(tree1(S, s) for S in "AFG") / 3 + .5 * NN[s]), PXD[s]) for s in SETS}
cnn = np.load(f"{W}/oof_gk_resnet34.a1_in1k_0.npy")[:, [0, 2, 1]]; nnk = np.load(f"{PC}/keep_oof_NNk.npy"); km = lambda S, b, s: np.load(f"{NV}/keep_oof_{S}_{b}D_{s}.npy")
QG = {s: (2 * km("A", "b2", s) + 2 * km("F", "b2", s) + km("A", "G", s) + km("F", "G", s) + cnn + nnk) / 8 for s in SETS}
cg = np.array(["left", "right", "stands in the middle"]); yg = np.asarray(gt["goalkeeper_direction"].values, dtype=str); yk = np.asarray(yk, dtype=str); N = len(gt)   # plain numpy strings (pandas string arrays do not broadcast)
pk = {s: cl[QK[s].argmax(1)] for s in SETS}; pg = {s: cg[QG[s].argmax(1)] for s in SETS}
ok_k = {s: (pk[s] == yk).astype(float) for s in SETS}; ok_g = {s: (pg[s] == yg).astype(float) for s in SETS}; sc = {s: .7 * ok_k[s] + .3 * ok_g[s] for s in SETS}
src = gt["source_id"].values; us = np.unique(src); ix = {u: np.where(src == u)[0] for u in us}; scode = pd.factorize(src)[0]

# ---------------- label-free rules (use only metadata a test row has) ----------------
foot = np.asarray(gt["preferred_foot"].str.lower().values, dtype=str); nat = np.where(foot == "right", "left", "right")       # a right-footer's natural side is the keeper's right = "left" in the labels
RULES = {"natural side (kick and keeper)": (nat, nat), "majority (left, left)": (np.full(N, "left"), np.full(N, "left")), "opposite of natural side": (np.where(nat == "left", "right", "left"),) * 2}
rule_sc = {nm: .7 * (k == yk) + .3 * (g == yg) for nm, (k, g) in RULES.items()}
PUB, PUB_KICK, PUB_KEEP, CHANCE = 0.83361, 289, 326, 0.52277; N1, N2 = 360, 540   # N6's public score (submitted 6 Nov, 22:53 UTC) = (7 x 289 + 3 x 326) / 3600; N4 scored 0.83611 = 289 / 329

pr(f"=== final_assess {time.strftime('%Y-%m-%d %H:%M')} | recipe N6 out of fold on {len(SETS)} partition sets, {N} labelled clips, {len(us)} sources")
pr("per set: kick right / keeper right / metric  " + " | ".join(f"{s} {int(ok_k[s].sum())} / {int(ok_g[s].sum())} / {sc[s].mean():.4f}" for s in SETS))
CVm = float(np.mean([sc[s].mean() for s in SETS])); CVk = float(np.mean([ok_k[s].mean() for s in SETS])); CVg = float(np.mean([ok_g[s].mean() for s in SETS]))
pr(f"mean over sets: kick {CVk:.4f} keeper {CVg:.4f} metric {CVm:.4f} | spread between sets (SD) {np.std([sc[s].mean() for s in SETS], ddof=1):.4f}")
pr("label-free rules on the same clips: " + " | ".join(f"{nm} kick {np.mean(RULES[nm][0] == yk):.4f} keeper {np.mean(RULES[nm][1] == yg):.4f} metric {rule_sc[nm].mean():.4f}" for nm in RULES)
   + f" | uniform random guessing: expected metric 0.3333 | class-frequency random guessing: expected {.7 * sum((np.mean(yk == c)) ** 2 for c in cl) + .3 * sum((np.mean(yg == c)) ** 2 for c in cg):.4f}")
a_ = [(a, b) for b in range(361) for a in range(361) if 7 * a + 3 * b == round(CHANCE * 3600)]; near = sorted(a_, key=lambda t: abs(t[0] - t[1]))[:3]
pr(f"chance.csv on the public board {CHANCE} = (7 kick + 3 keeper) / 3600 = {round(CHANCE * 3600)}; nearest balanced splits (kick, keeper right of 360): {near}. "
   f"The PDF quotes 0.52277 for its 'educated guess' baseline (kick and keeper to the kicker's natural side), so chance.csv is that rule, not coin flips (coin flips would score about 0.33).")

# ---------------- A1 exact tests against guessing ----------------
pr("\n== A1. exact one-sided binomial tests, out of fold (each set alone; a clip is right or wrong)")
for s in SETS:
    k_, g_ = int(ok_k[s].sum()), int(ok_g[s].sum())
    pr(f"   {s}: kick {k_}/{N} vs uniform 1/3 p = {binomtest(k_, N, 1 / 3, 'greater').pvalue:.1e}, vs the best label-free kick rule ({max(np.mean(r[0] == yk) for r in RULES.values()):.4f}) p = {binomtest(k_, N, max(np.mean(r[0] == yk) for r in RULES.values()), 'greater').pvalue:.1e}"
       f" | keeper {g_}/{N} vs the best label-free keeper rule ({max(np.mean(r[1] == yg) for r in RULES.values()):.4f}) p = {binomtest(g_, N, max(np.mean(r[1] == yg) for r in RULES.values()), 'greater').pvalue:.1e}")
pr("   (p printed as 0.0e+00 is below 1e-300, the floating-point floor)")

# ---------------- A2 permutation tests ----------------
pr("\n== A2. permutation tests: the predictions stay fixed, the labels are reshuffled (100,000 times over all clips; 20,000 times within each source)")
def perm_scores(s, B, within):
    out = np.empty(B); o = np.argsort(scode, kind="stable"); yk_, yg_ = yk, yg
    for b0 in range(0, B, 2000):
        nb = min(2000, B - b0)
        if within: order = np.argsort(scode[None, :] + rng.random((nb, N)), axis=1); idx = np.empty_like(order); idx[:, o] = order
        else: idx = np.argsort(rng.random((nb, N)), axis=1)
        out[b0:b0 + nb] = (.7 * (pk[s][None, :] == yk_[idx]) + .3 * (pg[s][None, :] == yg_[idx])).mean(1)
    return out
A2 = {}
for s in SETS[:2]:
    for within, B in ((False, 100000), (True, 20000)):
        z = perm_scores(s, B, within); obs = sc[s].mean(); A2[s, within] = z; hit = int((z >= obs).sum())
        tail = f"p < {1 / (B + 1):.0e} (no shuffle reached it)" if hit == 0 else f"p = {(hit + 1) / (B + 1):.1e}"
        pr(f"   {s} {'within sources' if within else 'all clips     '}: observed metric {obs:.4f} | shuffled labels mean {z.mean():.4f} SD {z.std():.4f}, max of {B:,} {z.max():.4f} | "
           f"observed is {(obs - z.mean()) / z.std():.0f} SD above | {tail}")
pr("   within-source shuffles keep each source's label mix, so a model that only knew the source's habits would keep its score; it falls to the same chance level")

# ---------------- A3 paired against the label-free rules; simulated private boards ----------------
pr("\n== A3. paired comparison with the label-free rules on the same clips (set 012, then all sets)")
for nm in RULES:
    for s in SETS[:1]:
        b = int(((ok_k[s] == 1) & (RULES[nm][0] != yk)).sum()); c = int(((ok_k[s] == 0) & (RULES[nm][0] == yk)).sum())
        pr(f"   vs {nm}: kick clips we get right and it gets wrong {b}, the reverse {c}, exact McNemar p = {binomtest(b, b + c, .5, 'greater').pvalue:.1e} | metric margin {sc[s].mean() - rule_sc[nm].mean():+.4f}")
sz = pm["source_id"].value_counts().values; big = np.array([u for u in us if len(ix[u]) >= 4])
flat = np.concatenate([ix[u] for u in big]); st_ = np.cumsum([0] + [len(ix[u]) for u in big])[:-1]; nb_ = np.array([len(ix[u]) for u in big])
SALL = np.stack([sc[s] for s in SETS]); RSC = np.stack([rule_sc[nm] for nm in RULES]); RB = 100000; CH = 5000; g_ = np.random.default_rng(1)
ours_pub, ours_prv, rule_prv = np.empty(RB), np.empty(RB), np.empty((len(RULES), RB))
for b0 in range(0, RB, CH):
    # simulated test sets: 82 sources with the real test sizes, drawn from labelled sources with >= 4 clips, clips drawn within each; random 360 / 540 split
    S = g_.integers(0, len(big), (CH, len(sz))); rep = np.repeat(S, sz, axis=1); idx = flat[st_[rep] + (g_.random(rep.shape) * nb_[rep]).astype(int)]
    idx = np.take_along_axis(idx, np.argsort(g_.random(idx.shape), axis=1), axis=1); w = (np.arange(b0, b0 + CH) % len(SETS))[:, None]      # each board scored with one of the four sets in turn
    ours_pub[b0:b0 + CH] = SALL[w, idx[:, :N1]].mean(1); ours_prv[b0:b0 + CH] = SALL[w, idx[:, N1:]].mean(1); rule_prv[:, b0:b0 + CH] = RSC[:, idx[:, N1:]].mean(2)
pr(f"   simulated private boards ({RB:,}; 540 clips each, test-sized sources): our private score mean {ours_prv.mean():.4f}, lowest {ours_prv.min():.4f}, 0.01% quantile {np.quantile(ours_prv, 1e-4):.4f}")
for j, nm in enumerate(RULES):
    rp = rule_prv[j]; m = ours_prv - rp
    pr(f"   vs {nm}: rule's private mean {rp.mean():.4f}, highest {rp.max():.4f} | our margin mean {m.mean():+.4f}, smallest {m.min():+.4f} | ours ahead on {np.mean(m > 0):.4%} of boards")
pr(f"   vs chance.csv as scored (0.52277 public): our lowest simulated private score {ours_prv.min():.4f} is {ours_prv.min() - CHANCE:+.4f} above it; normal tail estimate of P(private <= 0.52277) = {norm.cdf((CHANCE - ours_prv.mean()) / ours_prv.std()):.1e}")

# ---------------- A4 confidence vs accuracy ----------------
pr("\n== A4. does confidence carry information? accuracy by confidence fifth (set 012). A guessing model would be flat.")
for nm, Q, ok in (("kick", QK["012"], ok_k["012"]), ("keeper", QG["012"], ok_g["012"])):
    c = Q.max(1); q = pd.qcut(c, 5, labels=False, duplicates="drop"); acc = [ok[q == i].mean() for i in range(q.max() + 1)]
    pr(f"   {nm}: " + " | ".join(f"conf {c[q == i].mean():.2f} -> right {a:.3f}" for i, a in enumerate(acc)) + f" | Spearman(conf, right) {spearmanr(c, ok)[0]:.3f}, p {spearmanr(c, ok)[1]:.1e}")

# ---------------- A5 groups ----------------
pr("\n== A5. every subgroup, set 012: our metric vs the natural-side rule (n clips)")
fps = gt["fps"].values.astype(float); fb = np.select([fps < 20, fps < 27, fps < 35], ["15 fps", "24-26 fps", "30 fps"], "50-60 fps")
clk = set(pd.read_csv(f"{D}/public_position_clicks.csv")["clip"])
csrc = set(gt.loc[gt["clip"].isin(clk), "source_id"]); CAMd = pickle.load(open(f"{PC}/camera2.pkl", "rb")); phi = np.array([CAMd[c]["phi"] if c in CAMd else np.nan for c in gt["clip"]])
GR = {"fps": fb, "resolution": gt["resolution"].astype(str).values, "foot": foot, "gender": gt["gender"].values, "camera side": np.where(np.cos(np.radians(phi)) >= 0, "phi < 90", "phi >= 90"),
      "clicked source": np.where(np.isin(src, list(csrc)), "clicked", "unclicked"), "kick class": yk}
worst = []
for gname, g in GR.items():
    parts = []
    for v in pd.unique(g):
        m = g == v
        if m.sum() < 10: continue
        d = sc["012"][m].mean() - rule_sc["natural side (kick and keeper)"][m].mean(); worst.append((d, gname, v, int(m.sum()))); parts.append(f"{v} {sc['012'][m].mean():.3f} vs {rule_sc['natural side (kick and keeper)'][m].mean():.3f} (n {int(m.sum())})")
    pr(f"   {gname}: " + " | ".join(parts))
w = min(worst); pr(f"   smallest margin over all groups with n >= 10: {w[0]:+.3f} ({w[1]} = {w[2]}, n {w[3]})")
pr("   (kick class 'middle': the natural-side rule never calls middle, so that row compares keeper skill mostly)")

# ---------------- A6 public board ----------------
pr("\n== A6. the public board itself")
pk_c = np.mean([np.mean(RULES['natural side (kick and keeper)'][0] == yk)]); s_rule = rule_sc["natural side (kick and keeper)"].std()
pr(f"   ours {PUB} = kick {PUB_KICK}/360, keeper {PUB_KEEP}/360; chance.csv 0.52277. If our file were no better than chance.csv (per-clip score SD {s_rule:.3f}), the public score would be "
   f"0.52277 +- {s_rule / np.sqrt(N1):.4f}; ours is {(PUB - CHANCE) / (s_rule / np.sqrt(N1)):.0f} SD above, one-sided p {norm.sf((PUB - CHANCE) / (s_rule / np.sqrt(N1))):.0e}")
pr(f"   kick alone: {PUB_KICK}/360 against the natural-side rate {pk_c:.3f}: exact p = {binomtest(PUB_KICK, N1, pk_c, 'greater').pvalue:.1e}")

# ---------------- B1-B2 gap between the boards ----------------
pr("\n== B1. how far apart public (360) and private (540) can be with no overfitting at all (clips split at random, as the README says)")
s_all = float(np.mean([sc[s].std() for s in SETS])); gsd = s_all * np.sqrt(1 / N1 + 1 / N2)
pr(f"   per-clip score SD {s_all:.4f} -> SD(public - private) = {gsd:.4f}")
for d, nm in ((PUB * .01, f"1% of public ({PUB * .01:.4f})"), (.01, "1 point (0.01)"), (.02, "2 points"), (.03, "3 points")):
    pr(f"   chance |public - private| <= {nm}: {2 * norm.cdf(d / gsd) - 1:.1%} | chance private >= public - that: {norm.cdf(d / gsd):.1%}")
for d in (PUB * .01, .01):
    n_ = (norm.ppf(.975) * s_all / d) ** 2; pr(f"   to be 95% sure of a gap within {d:.4f}, both boards would need about {n_ * 2:,.0f} clips each (they have 360 and 540)")
gap = ours_pub - ours_prv
pr(f"\n== B2. simulated boards ({RB:,}, test-sized sources): gap SD {gap.std():.4f}, 5-95% [{np.percentile(gap, 5):+.4f}, {np.percentile(gap, 95):+.4f}] | "
   + " | ".join(f"|gap| <= {d:.4f}: {np.mean(np.abs(gap) <= d):.1%}" for d in (PUB * .01, .01, .02)))

# ---------------- B3 forecast ----------------
def comps(y, g):
    a = pd.DataFrame(dict(y=y, g=g)).groupby("g")["y"].agg(["size", "mean", "var"]).fillna(0); n, J = len(y), len(a); sw2 = ((a["size"] - 1) * a["var"]).sum() / (n - J)
    Q = (a["size"] * (a["mean"] - y.mean()) ** 2).sum(); return max((Q - (J - 1) * sw2) / (n - (a["size"] ** 2).sum() / n), 0.0), sw2
def forecast(m, pub_, sb2, sw2, s2, n_t, n_c, s_sys=0.0):
    tau2 = sb2 * (n_t ** 2).sum() / n_t.sum() ** 2 + sw2 / n_t.sum(); se2 = sb2 * (n_c ** 2).sum() / n_c.sum() ** 2 + sw2 / n_c.sum(); t2 = tau2 + se2 + s_sys ** 2; e2 = s2 * .6 / N1
    k = t2 / (t2 + e2); mp = m + k * (pub_ - m); vp = t2 * e2 / (t2 + e2); return 5 / 3 * mp - 2 / 3 * pub_, 5 / 3 * np.sqrt(vp), k
sb2, sw2 = np.mean([comps(sc[s], src) for s in SETS], 0); s2 = float(np.mean([sc[s].var() for s in SETS])); nc = np.array([len(ix[u]) for u in us])
mix = pd.Series(fb).value_counts(normalize=True); fpm = pm["fps"].astype(float).values; tmix = pd.Series(np.select([fpm < 20, fpm < 27, fpm < 35], ["15 fps", "24-26 fps", "30 fps"], "50-60 fps")).value_counts(normalize=True)
wts = np.array([tmix.get(v, 0) / mix[v] for v in fb]); CVt = float(np.mean([np.average(sc[s], weights=wts) for s in SETS]))
lowres = np.array(["640" in str(r) for r in gt["resolution"]]); lr_test = float(np.mean(["640" in str(r) for r in pm["resolution"]]))
CVr = float(np.mean([(1 - lr_test) * sc[s][~lowres].mean() + lr_test * sc[s][lowres].mean() for s in SETS])) if lowres.any() else CVm
EST = {"CV as is": CVm, "CV on the test fps mix": CVt, "CV on the test resolution mix": CVr}; m0 = float(np.mean(list(EST.values()))); sys0 = float(np.std(list(EST.values()), ddof=1))
pr(f"\n== B3. forecast of the private score given public {PUB} (the 2 Nov method; its backtest on 20,000 pseudo boards landed 80.3% / 95.4% inside the stated 80% / 95% ranges)")
pr("   level estimates: " + " | ".join(f"{k} {v:.4f}" for k, v in EST.items()) + f" -> mean {m0:.4f}, spread {sys0:.4f} | variance parts: between sources SD {np.sqrt(sb2):.4f}, within {np.sqrt(sw2):.4f}")
e_, s_, k_ = forecast(m0, PUB, sb2, sw2, s2, sz, nc, sys0)
pr(f"   private forecast {e_:.4f}, SD {s_:.4f} | 80% range {e_ - 1.2816 * s_:.4f} to {e_ + 1.2816 * s_:.4f} | 95% range {e_ - 1.96 * s_:.4f} to {e_ + 1.96 * s_:.4f}")
for d, nm in ((PUB * .01, "within 1% of public"), (.01, "within 1 point of public")):
    pr(f"   chance private is {nm} ({PUB - d:.4f} to {PUB + d:.4f}): {norm.cdf((PUB + d - e_) / s_) - norm.cdf((PUB - d - e_) / s_):.1%} | at least {PUB - d:.4f}: {norm.sf((PUB - d - e_) / s_):.1%}")
for c in (.9, .95, .99, .999): pr(f"   {c:.1%} chance private is at least {e_ + norm.ppf(1 - c) * s_:.4f}")
pr(f"   why the forecast sits below public: public is {PUB - m0:+.4f} above the level estimate ({(PUB - m0) / np.sqrt(s2 * .6 / N1):.1f} SD of the 360-clip noise). The 900 test clips' true level is "
   f"estimated as {k_:.0%} public + {1 - k_:.0%} CV, and since 900 x level = 360 x public + 540 x private, a lucky public leaves less for private")
for d_ in (0.0, -.01, -.02):
    e2_, s2_, _ = forecast(m0 + d_, PUB, sb2, sw2, s2, sz, nc, sys0); pr(f"   if the true level were {d_:+.2f} from the estimate: private {e2_:.4f} SD {s2_:.4f}, chance within 1 point of public {norm.cdf((PUB + .01 - e2_) / s2_) - norm.cdf((PUB - .01 - e2_) / s2_):.1%}")

# ---------------- B4 selection ----------------
pr("\n== B4. did picking the file on the public board inflate public? (public scores of our candidates, from the submission history)")
pr("   P4rkC, P4rkE, P4rkD, N4 all scored 0.83611 while differing from each other on 3-29 test kicks; N4 and N6 were chosen by rules written before their CV results (RULES_nov4.md, RULES_nov6.md), "
   "not by public scores. N6 then scored 0.83361 (its 7 changed test keepers lost 3 clips net on public) and was kept as the final file, by the rule: it is the LOWER of the two public scores, "
   "so the selected file's public score carries no selection lift beyond the luck measured in B3.")
pickle.dump(dict(CVm=CVm, est=EST, forecast=(e_, s_), gap_sd=gsd, ours_prv=ours_prv, sets=SETS), open(f"{PC}/final/assess.pkl", "wb"))
pr("\ndone")
