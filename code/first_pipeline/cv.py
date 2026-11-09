"""Stage 5: source-grouped CV of per-target gradient-boosted models on kinematic features, with mirror augmentation + TTA."""
import pickle, sys, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.model_selection import StratifiedGroupKFold

ROOT = r"C:\path\to\project-root"
D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
feats = pickle.load(open(f"{C}/feats.pkl", "rb"))
gt = pd.read_csv(f"{D}/train_ground_truth.csv")
gt = gt[gt["clip"].isin(feats)].reset_index(drop=True)
X = np.stack([feats[c][0] for c in gt["clip"]]); Xm = np.stack([feats[c][1] for c in gt["clip"]])
print("train clips with features", len(gt), "of 1300; dim", X.shape[1])
SW = {"left": "right", "right": "left", "middle": "middle", "stands in the middle": "stands in the middle"}
T = {"kick": ("kick_direction", True), "gk": ("goalkeeper_direction", True), "goal": ("goal_outcome", False)}
cols = {"all": slice(None), "kicker_only": np.r_[0:155, 255:268], "keeper_only": np.r_[155:268], "meta_only": np.r_[255:268]}
params = dict(max_iter=250, learning_rate=0.04, max_depth=3, min_samples_leaf=25, l2_regularization=2.0, max_features=0.5)


def cv(Xa, Xb, target, seeds=(0, 1)):
    col, swap = T[target]; y = gt[col].values; ym = np.array([SW[v] for v in y]) if swap else y
    classes = sorted(set(y)); grp = gt["source_id"].values; accs = []
    oof = np.zeros((len(seeds), len(y), len(classes)))
    for si, s in enumerate(seeds):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(Xa, y, grp):
            m = HGB(**params, random_state=s).fit(np.r_[Xa[tr], Xb[tr]], np.r_[y[tr], ym[tr]])
            p = m.predict_proba(Xa[te])
            pm_ = m.predict_proba(Xb[te])
            # mirrored sample's prediction maps back by swapping left/right columns
            idx = [list(m.classes_).index(SW[k]) if swap else i for i, k in enumerate(m.classes_)]
            pm_ = pm_[:, idx]
            oof[si, te] = (p + pm_) / 2
        accs.append((np.array(classes)[oof[si].argmax(1)] == y).mean())
    return np.mean(accs), oof.mean(0), classes


if __name__ == "__main__":
    base = {"kick": (gt["kick_direction"] == "left").mean(), "gk": (gt["goalkeeper_direction"] == "left").mean(), "goal": (gt["goal_outcome"] == "goal").mean()}
    print("majority:", {k: round(v, 3) for k, v in base.items()})
    for nm, cs in cols.items():
        r = {}
        for t in ("kick", "gk", "goal"):
            r[t], _, _ = cv(X[:, cs], Xm[:, cs], t)
        print(f"{nm:12s} kick {r['kick']:.3f} gk {r['gk']:.3f} goal {r['goal']:.3f} score {0.6 * r['kick'] + 0.25 * r['gk'] + 0.15 * r['goal']:.3f}", flush=True)
