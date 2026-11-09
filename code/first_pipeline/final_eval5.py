"""Out-of-fold evaluation of the FINAL v5 configuration (source-grouped 5-fold, 3 seeds, mirror aug + mirror TTA).
kick = mean(GBDT, ExtraTrees) on kin.pkl ; keeper = mean(GBDT, ExtraTrees on keepfeat.pkl, keeper CNN OOF) ; outcome = always goal."""
import pickle, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier as HGB, ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); grp = gt["source_id"].values; yk = gt["kick_direction"].values; yg = gt["goalkeeper_direction"].values; yo = gt["goal_outcome"].values
SW = {"left": "right", "right": "left", "middle": "middle", "stands in the middle": "stands in the middle"}


def load(path):
    K = pickle.load(open(path, "rb")); r = K["rows"]; nan = np.full(len(K["names"]), np.nan, np.float32)
    return np.stack([r.get(c, (nan, nan))[0] for c in gt["clip"]]), np.stack([r.get(c, (nan, nan))[1] for c in gt["clip"]])


def oof(mk, Xa, Xb, y, classes, seeds=(0, 1, 2)):
    ym = np.array([SW[v] for v in y]); P = np.zeros((len(y), len(classes)))
    for s in seeds:
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(Xa, y, grp):
            m = mk(s).fit(np.r_[Xa[tr], Xb[tr]], np.r_[y[tr], ym[tr]]); c_ = list(m.classes_); ix = [c_.index(SW[k]) for k in c_]
            p = (m.predict_proba(Xa[te]) + m.predict_proba(Xb[te])[:, ix]) / 2; P[te] += p[:, [c_.index(k) for k in classes]] / len(seeds)
    return P


hg = lambda s: HGB(random_state=s, max_iter=250, learning_rate=0.04, max_depth=4, min_samples_leaf=25, l2_regularization=2.0, max_features=0.5)
et = lambda leaf: (lambda s: make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(500, min_samples_leaf=leaf, max_features=0.3, n_jobs=-1, random_state=s)))
ck = ["left", "middle", "right"]; cg = ["left", "right", "stands in the middle"]
Xa, Xb = load(f"{C}/kin.pkl"); Ka, Kb = load(f"{C}/keepfeat.pkl")
Ph, Pe = oof(hg, Xa, Xb, yk, ck), oof(et(3), Xa, Xb, yk, ck); Pk = (Ph + Pe) / 2
Gh, Ge = oof(hg, Ka, Kb, yg, cg), oof(et(2), Ka, Kb, yg, cg)
cnn = np.load(f"{C}/oof_gk_resnet34.a1_in1k_0.npy")[:, [0, 2, 1]]; Pg = (Gh + Ge + cnn) / 3
a = lambda P, cl, y: (np.array(cl)[P.argmax(1)] == y).mean()
print(f"kick   : GBDT {a(Ph, ck, yk):.3f} | ExtraTrees {a(Pe, ck, yk):.3f} | mean {a(Pk, ck, yk):.3f} (left/right-only {(np.array(ck)[Pk.argmax(1)][yk != 'middle'] == yk[yk != 'middle']).mean():.3f})")
print(f"keeper : GBDT {a(Gh, cg, yg):.3f} | ExtraTrees {a(Ge, cg, yg):.3f} | CNN {a(cnn, cg, yg):.3f} | mean of 3 {a(Pg, cg, yg):.3f}")
go = (yo == "goal").mean(); print(f"outcome: always goal {go:.3f}")
print(f"ESTIMATED CV SCORE (v5): {0.6 * a(Pk, ck, yk) + 0.25 * a(Pg, cg, yg) + 0.15 * go:.4f}   [Kaggle public LB: v2 0.61555, v3 0.65555, v4 0.71555, v5 0.72250]")
