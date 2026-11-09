import pickle, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); yk, yg, yo = gt["kick_direction"].values, gt["goalkeeper_direction"].values, gt["goal_outcome"].values; src = gt["source_id"].values
cl = np.array(["left", "middle", "right"]); cg = np.array(["left", "right", "stands in the middle"]); cfg = [(12, 96), (15, 128), (20, 96), (20, 128), (30, 128)]
tr = lambda n: (np.load(f"{C}/base_hgb{n}.npy") + np.load(f"{C}/base_et{n}.npy")) / 2; bag = lambda st: sum(np.load(f"{C}/oof_seq2_{e}_{h}_b1_kc0_ax0_j0_st{st}.npy") for e, h in cfg) / 5
tA, tF, tG = tr(""), tr("_F"), tr("_G"); nEF, nG = bag("EF"), bag("G"); Pk = 0.5 * (tA + tF + tG) / 3 + 0.5 * (nEF + nG) / 2
kr, kf = np.load(f"{C}/keep_oof_rtmw.npy"), np.load(f"{C}/keep_oof_F.npy"); cnn = np.load(f"{C}/oof_gk_resnet34.a1_in1k_0.npy")[:, [0, 2, 1]]; Pg = (2 * kr + 2 * kf + cnn) / 5
pk, pg = cl[Pk.argmax(1)], cg[Pg.argmax(1)]; okk, okg = pk == yk, pg == yg; go = (yo == "goal").mean(); lr = yk != "middle"; sc = lambda k, g, o=go: 0.6 * k + 0.25 * g + 0.15 * o
print(f"CURRENT (v10 recipe, out-of-fold): kick {okk.mean():.3f} (L/R-only {(pk[lr] == yk[lr]).mean():.3f}) keeper {okg.mean():.3f} outcome {go:.3f} -> score {sc(okk.mean(), okg.mean()):.4f}")
print("\n=== 1. WHAT 0.80 REQUIRES ===")
for g, o in ((okg.mean(), go), (0.94, go), (0.94, 0.75), (0.96, 0.78)): print(f"  keeper {g:.3f}, outcome {o:.3f} -> kick accuracy needed {(0.80 - 0.25 * g - 0.15 * o) / 0.6:.3f}")
print("  kick score sensitivity: each +1 point of kick = +0.006 score; each +1 point of keeper = +0.0025; each +1 point of outcome = +0.0015")
lost = {"kick": 0.6 * (1 - okk.mean()), "keeper": 0.25 * (1 - okg.mean()), "outcome": 0.15 * (1 - go)}; tot = sum(lost.values()); print("  points lost:", {k: f"{v:.3f} ({v / tot:.0%})" for k, v in lost.items()}, f"| need to recover {1 - tot - 0.0:.3f} -> gap to 0.80 = {0.80 - (1 - tot):.3f}")
print("\n=== 2. ERROR TAXONOMY (kick) ===")
print(pd.crosstab(yk, pk).to_string()); n_err = (~okk).sum()
print(f"  errors {n_err}: left<->right swaps {((yk != 'middle') & (pk != 'middle') & ~okk).sum()} ({((yk != 'middle') & (pk != 'middle') & ~okk).sum() / n_err:.0%}) | true middle missed {((yk == 'middle') & ~okk).sum()} ({((yk == 'middle') & ~okk).sum() / n_err:.0%}) | non-middle called middle {((pk == 'middle') & (yk != 'middle')).sum()}")
models = {"trees RTMW": tA, "trees ViTPose-L": tF, "trees ViTPose-H": tG, "NN EF": nEF, "NN G": nG}; right = np.stack([cl[P.argmax(1)] == yk for P in models.values()], 1)
print(f"  models right on the same clip: all {right.all(1).mean():.3f} | none {(~right.any(1)).mean():.3f} (irreducible with these inputs) | some {(right.any(1) & ~right.all(1)).mean():.3f}; oracle 'any model right' accuracy {right.any(1).mean():.3f} vs blend {okk.mean():.3f}")
conf = Pk.max(1); wrong_conf = (~okk) & (conf > 0.8); print(f"  confident errors: {wrong_conf.sum()} clips with conf>0.8 wrong (of {(conf > 0.8).sum()} confident = {1 - okk[conf > 0.8].mean():.1%} error) | accuracy by confidence quintile: {[round(float(okk[(conf >= np.quantile(conf, a)) & (conf <= np.quantile(conf, b))].mean()), 3) for a, b in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1))]}")
print("\n=== 3. SEGMENTS (kick accuracy) ===")
K = pickle.load(open(f"{C}/kin.pkl", "rb")); names = list(K["names"]); X = np.stack([K["rows"][c][0] for c in gt["clip"]]); g = lambda n: X[:, names.index(n)]; av = pd.read_pickle(f"{C}/audit_final.pkl"); av = av[av.part == "train"].reset_index(drop=True)
d = pd.DataFrame(dict(ok=okk, side=np.abs(g("x_goalframe0")), fy=g("x_goalframe1"), kh=av["kh"].values, fps=g("meta1"), foot=gt["preferred_foot"].values, gender=gt["gender"].values, src=src, ball=av["ball_strong"].values, hip=g("hip_vel_0.30")))
for col, q in (("side", 3), ("kh", 3), ("fy", 3)): s = pd.qcut(d[col], q, duplicates="drop"); print(f"  {col:5s}:", [f"{int(r['size'])} clips {r['mean']:.3f}" for _, r in d.groupby(s, observed=True)["ok"].agg(["size", "mean"]).iterrows()])
print("  fps buckets:", [f"{int(r['size'])} clips {r['mean']:.3f}" for _, r in d.groupby(pd.cut(d.fps, [0, 26, 31, 55, 61]), observed=True)["ok"].agg(["size", "mean"]).iterrows()], "| foot R/L:", [f"{int(r['size'])} {r['mean']:.3f}" for _, r in d.groupby("foot")["ok"].agg(["size", "mean"]).iterrows()], "| M/F:", [f"{int(r['size'])} {r['mean']:.3f}" for _, r in d.groupby("gender")["ok"].agg(["size", "mean"]).iterrows()])
s = d.groupby("src")["ok"].agg(["size", "mean"]); s = s[s["size"] >= 8]; p = d.ok.mean(); print(f"  per-source accuracy (sources n>=8: {len(s)}): sd {s['mean'].std():.3f} vs binomial {np.sqrt((p * (1 - p) / s['size']).mean()):.3f}; worst 5:", s.sort_values("mean").head(5).round(2).to_dict("index"))
big = d.groupby("src").agg(n=("ok", "size"), err=("ok", lambda x: (~x).sum())).sort_values("err", ascending=False); print(f"  top-10 sources hold {big.err.head(10).sum() / big.err.sum():.0%} of errors (they hold {big.n.head(10).sum() / len(d):.0%} of clips)")
print("\n=== 4. KEEPER & OUTCOME ===")
print(f"  keeper errors {(~okg).sum()}: recall by class {dict(zip(sorted(set(yg)), [round(float((pg[yg == c] == c).mean()), 3) for c in sorted(set(yg))]))} | 'stands in the middle' = {(yg == 'stands in the middle').mean():.1%} of clips -> {0.25 * (yg == 'stands in the middle').mean():.4f} score at zero recall")
print(f"  outcome: always-goal {go:.3f}; no_goal rate when predicted kick side == predicted keeper side and kick conf>0.8: {(yo == 'no_goal')[(pk == pg) & (conf > .8)].mean():.3f} (n={((pk == pg) & (conf > .8)).sum()})")
print("\n=== 5. SCENARIOS (what each fix would be worth) ===")
for nm, k, gk in (("+2 points kick", okk.mean() + .02, okg.mean()), ("+4 points kick", okk.mean() + .04, okg.mean()), ("+6 points kick", okk.mean() + .06, okg.mean()), ("+6 kick, keeper 0.94", okk.mean() + .06, .94), ("middle recall 0.3 at precision 0.5 (est.)", okk.mean() + 0.008, okg.mean())): print(f"  {nm:42s} -> score {sc(k, gk):.4f}")
