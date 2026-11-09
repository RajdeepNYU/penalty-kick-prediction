import pickle, numpy as np, pandas as pd, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
warnings.filterwarnings("ignore")
ROOT = r"C:\path\to\project-root"
D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
gt = pd.read_csv(f"{D}/train_ground_truth.csv")
E = {c: pickle.load(open(f"{C}/emb/{c}.npy.pkl", "rb")) for c in gt["clip"]}
ok = [c for c in gt["clip"] if E[c] is not None]; gt = gt[gt["clip"].isin(ok)].reset_index(drop=True)
print("clips", len(gt))
SW = {"left": "right", "right": "left", "middle": "middle", "stands in the middle": "stands in the middle"}
kick = np.stack([E[c]["kick"] for c in gt["clip"]]).astype(np.float32)   # [N,4,2,1536]
keep = np.stack([E[c]["keep"] for c in gt["clip"]]).astype(np.float32)   # [N,3,2,1536]
foot = np.where(gt["preferred_foot"] == "right", 1.0, -1.0)[:, None]
grp = gt["source_id"].values


def make(arr, frames, flip):  # -> [N, F*1536]
    return arr[:, frames, flip, :].reshape(len(arr), -1)


def run(name, kfe, col, swap, ncomp=256, Cr=0.05, seeds=(0,)):
    y = gt[col].values; ym = np.array([SW[v] for v in y]); acc = []
    for s in seeds:
        pred = np.empty(len(y), object)
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(kfe(0), y, grp):
            A, B = kfe(0), kfe(1)
            sc = StandardScaler().fit(np.r_[A[tr], B[tr]]); pca = PCA(ncomp, random_state=0).fit(sc.transform(np.r_[A[tr], B[tr]]))
            f = lambda Z, fo: np.c_[pca.transform(sc.transform(Z)), fo * 3]
            m = LogisticRegression(C=Cr, max_iter=500).fit(np.r_[f(A[tr], foot[tr]), f(B[tr], -foot[tr])], np.r_[y[tr], ym[tr]])
            p = m.predict_proba(f(A[te], foot[te])); pm_ = m.predict_proba(f(B[te], -foot[te]))
            idx = [list(m.classes_).index(SW[k]) for k in m.classes_]
            pred[te] = m.classes_[((p + pm_[:, idx]) / 2).argmax(1)]
        acc.append((pred == y).mean())
    print(f"{name:34s} {col[:6]} acc {np.mean(acc):.3f}", flush=True)


for nm, fr in (("kicker last frame", [0]), ("kicker last 2", [0, 1]), ("kicker 4 frames", [0, 1, 2, 3])):
    run(nm, lambda fl, fr=fr: make(kick, fr, fl), "kick_direction", True)
for nm, fr in (("keeper last frame", [0]), ("keeper 3 frames", [0, 1, 2])):
    run(nm, lambda fl, fr=fr: make(keep, fr, fl), "goalkeeper_direction", True)
