import pickle, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); yk, yg, yo = gt["kick_direction"].values, gt["goalkeeper_direction"].values, gt["goal_outcome"].values
Pk = np.load(f"{C}/new_Pk.npy"); Pg = np.load(f"{C}/new_Pg.npy"); trees = np.load(f"{C}/new_trees.npy"); nn = np.load(f"{C}/new_nn.npy"); cl = np.array(["left", "middle", "right"]); pk = cl[Pk.argmax(1)]
pg = np.array(["left", "right", "stands in the middle"])[Pg.argmax(1)]; okk, okg = pk == yk, pg == yg; go = (yo == "goal").mean()
lost = {"kick": 0.6 * (1 - okk.mean()), "keeper": 0.25 * (1 - okg.mean()), "outcome": 0.15 * (1 - go)}; tot = sum(lost.values())
print("LOSS BUDGET (points lost of 1.0):", {k: f"{v:.3f} ({v / tot:.0%})" for k, v in lost.items()}, f"| total {tot:.3f} -> score {1 - tot:.3f}")
print("\nKICK confusion (rows true, cols predicted):"); print(pd.crosstab(yk, pk).to_string())
n_err = (~okk).sum(); mid_true = ((yk == "middle") & ~okk).sum(); swap = ((yk != "middle") & (pk != "middle") & ~okk).sum(); mid_pred_wrong = ((pk == "middle") & (yk != "middle")).sum()
print(f"kick errors {n_err}: true-middle missed {mid_true} ({mid_true / n_err:.0%}) | left<->right swaps {swap} ({swap / n_err:.0%}) | non-middle predicted middle {mid_pred_wrong} ({mid_pred_wrong / n_err:.0%})")
lr = yk != "middle"; print(f"ceilings: left/right-only acc {(pk[lr] == yk[lr]).mean():.3f}; if every middle were found (oracle) kick would be {((pk[lr] == yk[lr]).sum() + (~lr).sum()) / len(yk):.3f} (+{0.6 * (((pk[lr] == yk[lr]).sum() + (~lr).sum()) / len(yk) - okk.mean()):.3f} score); if left/right were perfect (middle as now) kick {((lr.sum()) + ((pk == 'middle') & ~lr).sum()) / len(yk):.3f}")
K = pickle.load(open(f"{C}/kin.pkl", "rb")); names = list(K["names"]); X = np.stack([K["rows"][c][0] for c in gt["clip"]]); g = lambda n: X[:, names.index(n)]
d = pd.DataFrame(dict(ok=okk, conf=Pk.max(1), dur=-g("meta2"), fps=g("meta1"), sideon=np.abs(g("x_goalframe0")), fy=g("x_goalframe1"), pl_leg=g("pl_shank_angvel"), ft=np.where(gt["preferred_foot"] == "right", 1, 0), gender=(gt["gender"] == "M").astype(int)))
av = pd.read_pickle(f"{C}/audit_final.pkl"); av = av[av.part == "train"].reset_index(drop=True); d["kh"] = av["kh"].values
print("\nKICK accuracy by segment (current pipeline):")
for col, q in (("conf", 4), ("dur", 4), ("sideon", 3), ("kh", 3)):
    s = pd.qcut(d[col], q, duplicates="drop"); print(f"  {col}:", d.groupby(s, observed=True)["ok"].agg(["size", "mean"]).round(3).apply(lambda r: f"[{int(r['size'])} clips {r['mean']:.3f}]", axis=1).tolist())
print("  preferred foot (right/left):", d.groupby("ft")["ok"].agg(["size", "mean"]).round(3).values.tolist(), "| gender M/F:", d.groupby("gender")["ok"].agg(["size", "mean"]).round(3).values.tolist())
print("  fps buckets:", d.groupby(pd.cut(d.fps, [0, 26, 31, 55, 61]), observed=True)["ok"].agg(["size", "mean"]).round(3).values.tolist())
print(f"\nmodel agreement: trees acc {(cl[trees.argmax(1)] == yk).mean():.3f}, NN {(cl[nn.argmax(1)] == yk).mean():.3f}; both right {np.mean((cl[trees.argmax(1)] == yk) & (cl[nn.argmax(1)] == yk)):.3f}; at least one right {np.mean((cl[trees.argmax(1)] == yk) | (cl[nn.argmax(1)] == yk)):.3f}")
conf = Pk.max(1); print("accuracy by confidence quintile:", [round(float(okk[(conf >= np.quantile(conf, a)) & (conf <= np.quantile(conf, b))].mean()), 3) for a, b in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1))])
print(f"\nKEEPER errors {(~okg).sum()}: by true class recall:", {c: round(float((pg[yg == c] == c).mean()), 3) for c in sorted(set(yg))}, "| 'stands in the middle' n =", (yg == "stands in the middle").sum())
