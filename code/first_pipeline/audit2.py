import pickle, numpy as np, pandas as pd, cv2, warnings, time
warnings.filterwarnings("ignore")
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); yk, yg, yo = gt["kick_direction"].values, gt["goalkeeper_direction"].values, gt["goal_outcome"].values
cov = pd.read_pickle(f"{C}/audit_cov.pkl"); cov = cov[cov.part == "train"].reset_index(drop=True)
M = pickle.load(open(f"{C}/oof_ens_lab2.pkl", "rb")); trees = (M["hgb259"] + M["extratrees"]) / 2
nn = sum(np.load(f"{C}/oof_seq_{e}_{h}_k0_s1_conv_b1.npy") for e, h in [(12, 96), (15, 128), (20, 96), (20, 128), (30, 128)]) / 5
Pk = 0.5 * trees + 0.5 * nn; pk = np.array(["left", "middle", "right"])[Pk.argmax(1)]
KL = pickle.load(open(f"{C}/oof_keep_lab.pkl", "rb")); Pg = (KL["new_hgb"] + KL["new_et"] + KL["cnn"]) / 3; pg = np.array(["left", "right", "stands in the middle"])[Pg.argmax(1)]
cov["okk"], cov["okg"], cov["goal"] = pk == yk, pg == yg, yo == "goal"; cov["lr_ok"] = np.where(yk != "middle", pk == yk, np.nan); cov["mid_rec"] = np.where(yk == "middle", pk == "middle", np.nan)
cov["healthy"] = (cov.kick_legs_ok >= .9) & (cov.keep_legs_ok >= .9) & (cov.ball_support >= 3) & (cov.kick_track == 1) & (cov.keep_track == 1)
# ---- image-quality stats (last frame + last 8 frames)
k_ = {}; t0 = time.time()
for i, c in enumerate(cov["clip"]):
    b = pickle.load(open(f"{C}/ballspot/{c}.pkl", "rb")); kb = pickle.load(open(f"{C}/kpts/{c}.pkl", "rb")); kk = pickle.load(open(f"{C}/kkeep/{c}.pkl", "rb"))
    cap = cv2.VideoCapture(f"{D}/train/{c}"); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    g = fr[-1]; H, W = g.shape
    def stats(box):
        x0, y0, x1, y1 = [int(v) for v in box]; x0, y0 = max(x0, 0), max(y0, 0); r = g[y0:y1, x0:x1]
        return (cv2.Laplacian(r, cv2.CV_64F).var(), r.std()) if r.size > 100 else (np.nan, np.nan)
    ks, gs = stats(kb["box"][-1]), stats(kk["box"][-1]); last = fr[-8:]
    dif = [np.abs(last[j].astype(np.float32) - last[j - 1].astype(np.float32)).mean() for j in range(1, len(last))]
    hL = max(kb["box"][-1][3] - kb["box"][-1][1], 20); M2 = b["M"][-2] if len(b["M"]) > 1 else np.eye(3); ctr = np.array([W / 2, H / 2, 1.0]); p = M2 @ ctr; pan = np.hypot(*(p[:2] / p[2] - ctr[:2])) / hL
    k_[c] = dict(k_sharp=ks[0], k_contrast=ks[1], g_sharp=gs[0], g_contrast=gs[1], dup_frac=float(np.mean(np.array(dif) < 0.3)), pan=pan, width=W, frames=len(fr))
    if i % 300 == 0: print(i, f"{time.time() - t0:.0f}s", flush=True)
q = pd.DataFrame.from_dict(k_, orient="index"); q["clip"] = q.index; cov = cov.merge(q, on="clip", suffixes=("", "_q")); cov.to_pickle(f"{C}/audit2.pkl")
def table(mask, name):
    s = cov[mask]; k, g, o = s.okk.mean(), s.okg.mean(), s.goal.mean()
    print(f"  {name:34s} n={len(s):4d} ({len(s) / len(cov):4.0%}) | kick {k:.3f} (L/R-only {s.lr_ok.mean():.3f}, middle recall {s.mid_rec.mean():.2f}) | keeper {g:.3f} | outcome(always goal) {o:.3f} | SCORE {0.6 * k + 0.25 * g + 0.15 * o:.3f}")
print("\n=== all metrics, healthy vs not (out-of-fold, 1300 training clips) ==="); table(cov.healthy | ~cov.healthy, "ALL"); table(cov.healthy, "HEALTHY (all detections ok)"); table(~cov.healthy, "NOT healthy")
fl = {"ball support<3": cov.ball_support < 3, "kicker legs weak": cov.kick_legs_ok < .9, "keeper legs weak": cov.keep_legs_ok < .9, "kicker track<13": cov.kick_track < 1, "keeper track<13": cov.keep_track < 1}
n_fail = sum(v.astype(int) for v in fl.values()); print("\n=== why clips are unhealthy (flags overlap) ===")
for k, v in fl.items(): print(f"  {k:20s} {v.sum():4d} clips ({v.mean():.0%}); only this one: {((v) & (n_fail == 1)).sum()}")
print(f"  exactly 1 flag: {(n_fail == 1).sum()} | 2 flags: {(n_fail == 2).sum()} | 3+ flags: {(n_fail >= 3).sum()}")
print("\n=== by number of failed detections ==="); [table(n_fail == n if n < 2 else n_fail >= 2, f"{n if n < 2 else '2+'} failed detection(s)") for n in (0, 1, 2)]
print("\n=== are failures driven by clip quality? (unhealthy rate by tertile of each factor) ===")
for col in ("kh", "gh", "k_sharp", "g_sharp", "k_contrast", "pan", "dup_frac", "width", "frames"):
    v = cov[col]; qs = v.quantile([0, 1 / 3, 2 / 3, 1]).values
    if len(set(qs)) < 4: print(f"  {col:11s} (few distinct values) unhealthy rate by value:", cov.groupby(v.round(2))["healthy"].apply(lambda s: round(1 - s.mean(), 3)).head(6).to_dict()); continue
    out = [f"[{qs[i]:.2f},{qs[i + 1]:.2f}] unhealthy {1 - cov[(v >= qs[i]) & (v <= qs[i + 1])].healthy.mean():.2f} kick {cov[(v >= qs[i]) & (v <= qs[i + 1])].okk.mean():.3f}" for i in range(3)]
    print(f"  {col:11s}", " | ".join(out))
print("\nduplicate-frame share (consecutive frames ~identical) among last 8 frames:", f"clips with >=25% duplicate pairs: {(cov.dup_frac >= .25).mean():.2%}; kick acc there {cov[cov.dup_frac >= .25].okk.mean():.3f} vs {cov[cov.dup_frac < .25].okk.mean():.3f}")
