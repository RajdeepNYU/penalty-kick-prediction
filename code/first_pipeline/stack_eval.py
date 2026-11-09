"""Learned ensemble of the base models (out-of-fold probabilities) with source-grouped nested CV. Mirror-symmetric logistic stacker + middle-rebalancing rule."""
import pickle, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier as HGB, ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); grp = gt["source_id"].values; yk = gt["kick_direction"].values; cl = np.array(["left", "middle", "right"]); lab = {"left": 0, "middle": 1, "right": 2}; y = np.array([lab[v] for v in yk]); ym = np.where(y == 1, 1, 2 - y)
SW = {"left": "right", "right": "left", "middle": "middle"}
if not os.path.exists(f"{C}/base_hgb.npy"):
    K = pickle.load(open(f"{C}/kin.pkl", "rb")); r = K["rows"]; Xa = np.stack([r[c][0] for c in gt["clip"]]); Xb = np.stack([r[c][1] for c in gt["clip"]]); ys = np.array([SW[v] for v in yk])
    def oof(mk):
        P = np.zeros((len(yk), 3))
        for s in (0, 1, 2):
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(Xa, yk, grp):
                m = mk(s).fit(np.r_[Xa[tr], Xb[tr]], np.r_[yk[tr], ys[tr]]); c_ = list(m.classes_); ix = [c_.index(SW[k]) for k in c_]
                p = (m.predict_proba(Xa[te]) + m.predict_proba(Xb[te])[:, ix]) / 2; P[te] += p[:, [c_.index(k) for k in ["left", "middle", "right"]]] / 3
        return P
    np.save(f"{C}/base_hgb.npy", oof(lambda s: HGB(random_state=s, max_iter=250, learning_rate=0.04, max_depth=4, min_samples_leaf=25, l2_regularization=2.0, max_features=0.5)))
    np.save(f"{C}/base_et.npy", oof(lambda s: make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(500, min_samples_leaf=3, max_features=0.3, n_jobs=-1, random_state=s))))
base = {"hgb": np.load(f"{C}/base_hgb.npy"), "et": np.load(f"{C}/base_et.npy")}
for e, h in [(12, 96), (15, 128), (20, 96), (20, 128), (30, 128)]: base[f"nn{e}_{h}"] = np.load(f"{C}/oof_seq_{e}_{h}_k0_s1_conv_b1.npy")
for extra in sorted(f for f in os.listdir(C) if f.startswith("oof_seqX_")): base[extra[9:-4]] = np.load(f"{C}/{extra}")
acc = lambda P: (P.argmax(1) == y).mean(); names = list(base)
for k, P in base.items(): print(f"  base {k:14s} {acc(P):.3f}")
trees = (base["hgb"] + base["et"]) / 2; nns = [k for k in names if k.startswith("nn") or k.startswith("X")]; nn = sum(base[k] for k in nns) / len(nns)
print(f"\ncurrent 50/50 (trees avg, NN bag of {len(nns)}): {acc(0.5 * trees + 0.5 * nn):.3f}")
for w in (0.3, 0.4, 0.5, 0.6, 0.7): print(f"   NN weight {w}: {acc((1 - w) * trees + w * nn):.3f}", end=" |")
print()
def feats(P_list):                                           # mirror-equivariant features: antisymmetric left-vs-right log-odds, symmetric middle log-odds
    out = []
    for P in P_list:
        P = np.clip(P, 1e-4, 1); out += [np.log(P[:, 0] / P[:, 2]), np.log(P[:, 1] / np.sqrt(P[:, 0] * P[:, 2]))]
    return np.stack(out, 1)
def mirror(F): G = F.copy(); G[:, 0::2] = -G[:, 0::2]; return G
def stack_cv(models, C_=1.0, seeds=(0, 1, 2)):
    F = feats([base[k] for k in models]); out = np.zeros((len(y), 3))
    for s in seeds:
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=100 + s).split(F, y, grp):
            m = LogisticRegression(C=C_, max_iter=2000).fit(np.r_[F[tr], mirror(F[tr])], np.r_[y[tr], ym[tr]]); p = m.predict_proba(F[te]); pm_ = m.predict_proba(mirror(F[te]))[:, [2, 1, 0]]; out[te] += (p + pm_) / 2 / len(seeds)
    return out
print("\nLEARNED STACKERS (mirror-symmetric logistic regression on log-odds features):")
grp_models = {"2 groups (trees avg, nn avg)": None}
Pl = {"hgb+et+nn-bag": ["hgb", "et", "NNBAG"], "all base models": names}
for nm, ms in Pl.items():
    if "NNBAG" in ms: base["NNBAG"] = nn
    for C_ in (0.03, 0.1, 1.0): print(f"  {nm:22s} C={C_:<5}: {acc(stack_cv(ms, C_)):.3f}")
best = stack_cv(["hgb", "et", "NNBAG"], 0.1)
print("\nMIDDLE rebalancing on the 50/50 blend and on the stacker (class = argmax(pL, kappa*pM, pR)):")
B = 0.5 * trees + 0.5 * nn
for name, P in (("50/50 blend", B), ("stacker", best)):
    res = []
    for kap in (0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0):
        Q = P.copy(); Q[:, 1] *= kap; pr = Q.argmax(1); res.append(f"k={kap}: {np.mean(pr == y):.3f} (mid pred {np.mean(pr == 1):.2f}, recall {np.mean(pr[y == 1] == 1):.2f})")
    print(f"  {name}: " + " | ".join(res))
