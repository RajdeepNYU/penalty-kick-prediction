"""Diversity test: non-tree learners (L2 logistic, RBF-SVM, MLP) on the kinematic features of each stream; does adding them to the v10 blend raise OOF kick accuracy?"""
import pickle, numpy as np, pandas as pd, warnings, sys
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); yk = gt["kick_direction"].values; grp = gt["source_id"].values; cl = np.array(["left", "middle", "right"]); SW = {"left": "right", "right": "left", "middle": "middle"}; ym = np.array([SW[v] for v in yk])
acc = lambda Q: (cl[Q.argmax(1)] == yk).mean(); lr = yk != "middle"; LR = lambda Q: (cl[Q.argmax(1)][lr] == yk[lr]).mean()
def feats(path, keep):
    r = pickle.load(open(f"{C}/{path}.pkl", "rb"))["rows"]; nan = np.full(259, np.nan, np.float32); a = np.stack([r.get(c, (nan, nan))[0] for c in gt["clip"]]); b = np.stack([r.get(c, (nan, nan))[1] for c in gt["clip"]])
    if keep is None: keep = ~np.isnan(a).all(0)
    return a[:, keep], b[:, keep]
def oof(mk, Xa, Xb, seeds=1):
    med = np.nanmedian(np.r_[Xa, Xb], 0); Xa = np.where(np.isnan(Xa), med, Xa); Xb = np.where(np.isnan(Xb), med, Xb); P = np.zeros((len(yk), 3))
    for s in range(seeds):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(Xa, yk, grp):
            m = mk(s).fit(np.r_[Xa[tr], Xb[tr]], np.r_[yk[tr], ym[tr]]); c_ = list(m.classes_); ix = [c_.index(SW[k]) for k in c_]; p = (m.predict_proba(Xa[te]) + m.predict_proba(Xb[te])[:, ix]) / 2; P[te] += p[:, [c_.index(k) for k in cl]] / seeds
    return P
qt = lambda: QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=0)
M = {"logreg": lambda s: make_pipeline(qt(), LogisticRegression(C=0.01, max_iter=400)),
     "svm": lambda s: make_pipeline(qt(), SVC(C=1.0, gamma="scale", probability=True, random_state=s)),
     "mlp": lambda s: make_pipeline(qt(), MLPClassifier((128,), alpha=3.0, max_iter=300, early_stopping=False, random_state=s))}
tr_ = lambda n: (np.load(f"{C}/base_hgb{n}.npy") + np.load(f"{C}/base_et{n}.npy")) / 2; cfg = [(12, 96), (15, 128), (20, 96), (20, 128), (30, 128)]; bag = lambda st: sum(np.load(f"{C}/oof_seq2_{e}_{h}_b1_kc0_ax0_j0_st{st}.npy") for e, h in cfg) / 5
base = 0.5 * (tr_("") + tr_("_F") + tr_("_G")) / 3 + 0.5 * (bag("EF") + bag("G")) / 2; print(f"v10 blend: {acc(base):.3f} (L/R {LR(base):.3f})")
streams = {"A": ("kin", None, ""), "F": ("kinF", np.load(f"{C}/kinF_keep.npy"), "_F"), "G": ("kinG", np.load(f"{C}/kinG_keep.npy"), "_G")}
res = {}
for st, (p, k, sfx) in streams.items():
    Xa, Xb = feats(p, k)
    for nm, mk in M.items():
        P = oof(mk, Xa, Xb); res[(st, nm)] = P; np.save(f"{C}/div_{nm}{sfx or '_A'}.npy", P); print(f"  {st} {nm:7s}: {acc(P):.3f} (L/R {LR(P):.3f})", flush=True)
for nm in M:
    D_ = sum(res[(s, nm)] for s in streams) / 3
    for w in (0.2, 0.33): B = (1 - w) * base + w * D_; print(f"blend + {nm} (3 streams, w={w}): {acc(B):.3f} (L/R {LR(B):.3f})")
Dall = sum(res.values()) / len(res)
for w in (0.2, 0.33): B = (1 - w) * base + w * Dall; print(f"blend + all-nontree (w={w}): {acc(B):.3f} (L/R {LR(B):.3f})")
