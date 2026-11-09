"""Learning curve: ET on ViTPose-L kinematic features, train on a fraction of each training fold (source-grouped CV), 3 partitions."""
import pickle, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); yk = gt["kick_direction"].values; grp = gt["source_id"].values; cl = np.array(["left", "middle", "right"]); SW = {"left": "right", "right": "left", "middle": "middle"}; ym = np.array([SW[v] for v in yk])
r = pickle.load(open(f"{C}/kinM.pkl", "rb"))["rows"]; nan = np.full(259, np.nan, np.float32); Xa = np.stack([r.get(c, (nan, nan))[0] for c in gt["clip"]]); Xb = np.stack([r.get(c, (nan, nan))[1] for c in gt["clip"]])
rng = np.random.RandomState(0)
for frac in (0.25, 0.5, 0.75, 1.0):
    acc = []
    for s in range(3):
        P = np.zeros((len(yk), 3))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(Xa, yk, grp):
            ug = np.unique(grp[tr]); keepg = rng.choice(ug, max(2, int(len(ug) * frac)), replace=False); tr = tr[np.isin(grp[tr], keepg)]
            m = make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(400, min_samples_leaf=3, max_features=0.3, n_jobs=-1, random_state=s)).fit(np.r_[Xa[tr], Xb[tr]], np.r_[yk[tr], ym[tr]])
            c_ = list(m.classes_); ix = [c_.index(SW[k]) for k in c_]; p = (m.predict_proba(Xa[te]) + m.predict_proba(Xb[te])[:, ix]) / 2; P[te] = p[:, [c_.index(k) for k in cl]]
        acc.append((cl[P.argmax(1)] == yk).mean())
    print(f"train fraction {frac:.2f} (~{int(frac * 0.8 * len(yk))} clips): ET kick acc {np.mean(acc):.3f}", flush=True)
