import pickle, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); yk, yg, yo = gt["kick_direction"].values, gt["goalkeeper_direction"].values, gt["goal_outcome"].values
cov = pd.read_pickle(f"{C}/audit_cov.pkl"); cov = cov[cov.part == "train"].reset_index(drop=True); assert (cov["clip"].values == gt["clip"].values).all()
M = pickle.load(open(f"{C}/oof_ens_lab2.pkl", "rb")); trees = (M["hgb259"] + M["extratrees"]) / 2
nn = sum(np.load(f"{C}/oof_seq_{e}_{h}_k0_s1_conv_b1.npy") for e, h in [(12, 96), (15, 128), (20, 96), (20, 128), (30, 128)]) / 5
Pk = 0.5 * trees + 0.5 * nn; cl = np.array(["left", "middle", "right"]); pk = cl[Pk.argmax(1)]
KL = pickle.load(open(f"{C}/oof_keep_lab.pkl", "rb")); Pg = (KL["new_hgb"] + KL["new_et"] + KL["cnn"]) / 3; pg = np.array(["left", "right", "stands in the middle"])[Pg.argmax(1)]
okk, okg = pk == yk, pg == yg; go = (yo == "goal").mean()
print(f"OOF: kick {okk.mean():.3f} keeper {okg.mean():.3f} outcome {go:.3f} -> score {0.6 * okk.mean() + 0.25 * okg.mean() + 0.15 * go:.4f}")
lost = {"kick": 0.6 * (1 - okk.mean()), "keeper": 0.25 * (1 - okg.mean()), "outcome": 0.15 * (1 - go)}; tot = sum(lost.values())
print("points lost:", {k: f"{v:.3f} ({v / tot:.0%})" for k, v in lost.items()}, f"| total {tot:.3f}")
ck = pd.read_csv(f"{D}/public_position_clicks.csv").set_index("clip"); tracks = pickle.load(open(f"{C}/tracks.pkl", "rb"))
def hit(c, who, key):
    if c not in ck.index: return np.nan
    w, h = pickle.load(open(f"{C}/det/{c}.pkl", "rb"))["wh"]; r = ck.loc[c]; p = (r[f"{who}_x_norm"] * w, r[f"{who}_y_norm"] * h); bx = tracks[c][key]
    return float(bx[0] <= p[0] <= bx[2] and bx[1] <= p[1] <= bx[3])
cov["kick_pick"] = [hit(c, "kicker", "kick_a") for c in cov["clip"]]; cov["keep_pick"] = [hit(c, "keeper", "keep_a") for c in cov["clip"]]
cov["okk"], cov["okg"] = okk, okg
def seg(name, mask, target="okk", sub=None):
    sub = cov[target].notna() if sub is None else (sub & cov[target].notna())
    a, b = cov[mask & sub], cov[~mask & sub]
    if len(a) and len(b): print(f"  {name:46s} yes: n={len(a):4d} acc {a[target].mean():.3f} | no: n={len(b):4d} acc {b[target].mean():.3f}   (diff {a[target].mean() - b[target].mean():+.3f})")
print("\nKICK accuracy vs detection factors"); seg("kicker box right (clicked clips only)", cov.kick_pick == 1, "okk", cov.kick_pick.notna()); seg("kicker legs confident >=90% frames", cov.kick_legs_ok >= .9)
seg("foot points (toes/heels) confident >=90%", cov.kick_foot_ok >= .9); seg("ball supported by >=3 frames", cov.ball_support >= 3); seg("ball supported by >=6 frames", cov.ball_support >= 6)
seg("keeper legs confident >=90% frames", cov.keep_legs_ok >= .9); seg("kicker box height > 80 px", cov.kh > 80)
allok = (cov.kick_legs_ok >= .9) & (cov.keep_legs_ok >= .9) & (cov.ball_support >= 3) & (cov.kick_track == 1) & (cov.keep_track == 1); seg("ALL detections healthy", allok)
print("\nKEEPER accuracy vs detection factors"); seg("keeper box right (clicked clips only)", cov.keep_pick == 1, "okg", cov.keep_pick.notna()); seg("keeper legs confident >=90% frames", cov.keep_legs_ok >= .9, "okg"); seg("keeper tracked in all 13 frames", cov.keep_track == 1, "okg"); seg("ball supported by >=3 frames", cov.ball_support >= 3, "okg")
# cluster (source) bootstrap
src = gt["source_id"].values; us = np.unique(src); idx = {s: np.where(src == s)[0] for s in us}; rng = np.random.RandomState(0); sc_ = []; kk_ = []
for _ in range(3000):
    pick = np.concatenate([idx[s] for s in rng.choice(us, len(us))]); kk_.append(okk[pick].mean()); sc_.append(0.6 * okk[pick].mean() + 0.25 * okg[pick].mean() + 0.15 * (yo[pick] == "goal").mean())
print(f"\nsource-cluster bootstrap 95% CI: kick {np.percentile(kk_, 2.5):.3f}-{np.percentile(kk_, 97.5):.3f} | score {np.percentile(sc_, 2.5):.3f}-{np.percentile(sc_, 97.5):.3f} (point {0.6 * okk.mean() + 0.25 * okg.mean() + 0.15 * go:.3f})")
