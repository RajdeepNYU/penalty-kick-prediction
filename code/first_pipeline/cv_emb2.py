import numpy as np, pandas as pd, warnings, sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
warnings.filterwarnings("ignore")
ROOT = r"C:\path\to\project-root"
D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
z = np.load(f"{C}/align_emb.npz"); idx = {c: i for i, c in enumerate(z["clips"])}
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); ii = [idx[c] for c in gt["clip"]]
kick = z["kick"][ii].astype(np.float32); keep = z["keep"][ii].astype(np.float32)   # [N,F,2,1536]
SW = {"left": "right", "right": "left", "middle": "middle", "stands in the middle": "stands in the middle"}
foot = np.where(gt["preferred_foot"] == "right", 1.0, -1.0)[:, None]; grp = gt["source_id"].values


def run(name, arr, frames, col, ncomp=128, Cr=0.02, seed=0):
    y = gt[col].values; ym = np.array([SW[v] for v in y]); pred = np.empty(len(y), object)
    A = arr[:, frames, 0].reshape(len(arr), -1); B = arr[:, frames, 1].reshape(len(arr), -1)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(A, y, grp):
        sc = StandardScaler().fit(np.r_[A[tr], B[tr]]); pca = PCA(ncomp, random_state=0).fit(sc.transform(np.r_[A[tr], B[tr]]))
        f = lambda Z, fo: np.c_[pca.transform(sc.transform(Z)), fo * 3]
        m = LogisticRegression(C=Cr, max_iter=500).fit(np.r_[f(A[tr], foot[tr]), f(B[tr], -foot[tr])], np.r_[y[tr], ym[tr]])
        p = m.predict_proba(f(A[te], foot[te])); pm_ = m.predict_proba(f(B[te], -foot[te]))
        ix = [list(m.classes_).index(SW[k]) for k in m.classes_]
        pred[te] = m.classes_[((p + pm_[:, ix]) / 2).argmax(1)]
    print(f"{name:30s} nc={ncomp} C={Cr}  {col[:6]} acc {(pred == y).mean():.3f}", flush=True)


for fr, nm in (([0], "last"), ([0, 1], "last2"), ([0, 2, 4], "0,.06,.15"), ([0, 1, 2, 3, 4, 5], "all6")):
    run("kicker aligned " + nm, kick, fr, "kick_direction")
for nc, Cr in ((64, 0.02), (256, 0.02), (128, 0.005), (128, 0.1)):
    run("kicker aligned last2", kick, [0, 1], "kick_direction", nc, Cr)
for fr, nm in (([0], "last"), ([0, 1], "0,.06"), ([0, 1, 2, 3], "all4")):
    run("keeper aligned " + nm, keep, fr, "goalkeeper_direction")
