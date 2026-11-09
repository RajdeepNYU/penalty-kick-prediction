"""TabPFN as an additional tabular learner on the ViTPose-large kinematic features (mirror-augmented, source-grouped CV, 1 seed first). Compares with ET/HGB baselines and the blend."""
import pickle, sys, time, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedGroupKFold
from tabpfn import TabPFNClassifier
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); yk = gt["kick_direction"].values; grp = gt["source_id"].values; cl = np.array(["left", "middle", "right"]); SW = {"left": "right", "right": "left", "middle": "middle"}; ym = np.array([SW[v] for v in yk])
tag = sys.argv[1] if len(sys.argv) > 1 else "F"; path = {"A": "kin", "F": "kinF", "G": "kinG", "H": "kinH"}[tag]; keep = np.load(f"{C}/{path}_keep.npy") if tag != "A" else None
r = pickle.load(open(f"{C}/{path}.pkl", "rb"))["rows"]; nan = np.full(259, np.nan, np.float32); Xa = np.stack([r.get(c, (nan, nan))[0] for c in gt["clip"]]); Xb = np.stack([r.get(c, (nan, nan))[1] for c in gt["clip"]])
if keep is None: keep = ~np.isnan(Xa).all(0)
Xa, Xb = Xa[:, keep], Xb[:, keep]; med = np.nanmedian(np.r_[Xa, Xb], 0); Xa = np.where(np.isnan(Xa), med, Xa); Xb = np.where(np.isnan(Xb), med, Xb); print("features", Xa.shape)
P = np.zeros((len(yk), 3)); t0 = time.time(); seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 1
for s in range(seeds):
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(Xa, yk, grp):
        m = TabPFNClassifier(n_estimators=8, random_state=s, ignore_pretraining_limits=True).fit(np.r_[Xa[tr], Xb[tr]], np.r_[yk[tr], ym[tr]]); c_ = list(m.classes_); ix = [c_.index(SW[k]) for k in c_]
        p = (m.predict_proba(Xa[te]) + m.predict_proba(Xb[te])[:, ix]) / 2; P[te] += p[:, [c_.index(k) for k in cl]] / seeds
    print("seed", s, "done", f"{time.time() - t0:.0f}s", flush=True)
acc = lambda Q: (cl[Q.argmax(1)] == yk).mean(); lr = yk != "middle"; sfx = "" if tag == "A" else "_" + tag
np.save(f"{C}/base_tabpfn{sfx}.npy", P)
T = (np.load(f"{C}/base_hgb{sfx}.npy") + np.load(f"{C}/base_et{sfx}.npy")) / 2
print(f"TabPFN {tag}: kick {acc(P):.3f} (L/R {(cl[P.argmax(1)][lr] == yk[lr]).mean():.3f}) | trees {acc(T):.3f} | trees+TabPFN {acc((T + P) / 2):.3f}")
