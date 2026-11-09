import numpy as np, sys, warnings
warnings.filterwarnings("ignore")
sys.argv = ["x"]
from cv import X, Xm, gt, SW, C
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.model_selection import StratifiedGroupKFold
grp = gt["source_id"].values
KP = list(np.r_[0:90]) + [261]; ALL = list(range(268))
base = dict(max_iter=250, learning_rate=0.04, max_depth=3, min_samples_leaf=25, l2_regularization=2.0, max_features=0.5)


def oof(cfg, cols, col, seeds=(0, 1, 2)):
    y = gt[col].values; ym = np.array([SW[v] for v in y]); Xa, Xb = X[:, cols], Xm[:, cols]; classes = sorted(set(y)); P = np.zeros((len(y), len(classes)))
    for s in seeds:
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(Xa, y, grp):
            m = HGB(random_state=s, **{**base, **cfg}).fit(np.r_[Xa[tr], Xb[tr]], np.r_[y[tr], ym[tr]]); cls = list(m.classes_); ix = [cls.index(SW[k]) for k in cls]
            P[te] += (m.predict_proba(Xa[te]) + m.predict_proba(Xb[te])[:, ix]) / 2 / len(seeds)
    return P, classes


yk, yg = gt["kick_direction"].values, gt["goalkeeper_direction"].values
accs = lambda P, cl, y: (np.array(cl)[P.argmax(1)] == y).mean()
print("=== KICK ===")
K = {n: oof(c, KP, "kick_direction") for n, c in {"d3it250": {}, "d4it250": {"max_depth": 4}, "d3it500": {"max_iter": 500}}.items()}
for n, (P, cl) in K.items(): print(f"  {n:10s} {accs(P, cl, yk):.3f}")
Pk = sum(P for P, _ in K.values()) / 3; cl_k = K["d3it250"][1]; ak = accs(Pk, cl_k, yk); print(f"  bag of 3   {ak:.3f}")
print("=== KEEPER ===")
G = {n: oof(c, ALL, "goalkeeper_direction") for n, c in {"d4it250": {"max_depth": 4}, "d3it500": {"max_iter": 500}}.items()}
for n, (P, cl) in G.items(): print(f"  {n:10s} {accs(P, cl, yg):.3f}")
Pg = sum(P for P, _ in G.values()) / 2; cl_g = G["d4it250"][1]; print(f"  GBDT bag   {accs(Pg, cl_g, yg):.3f}")
cnn = np.load(f"{C}/oof_gk_cnn30.npy")[:, [0, 2, 1]]
print(f"  CNN(30ep)  {accs(cnn, cl_g, yg):.3f}")
res = {}
for w in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
    res[w] = accs((1 - w) * Pg + w * cnn, cl_g, yg); print(f"  blend w_cnn={w}: {res[w]:.3f}")
for w in (0.5,):
    ag = res[w]
    print(f"\nESTIMATED CV SCORE (kick bag, keeper blend w={w}, goal=always goal 0.722): {0.6 * ak + 0.25 * ag + 0.15 * 0.7223:.4f}")
    print(f"  per-target: kick {ak:.3f} keeper {ag:.3f}; old (v2): kick 0.531 keeper 0.788 -> 0.6247")
np.save(f"{C}/oof_final_kick.npy", Pk); np.save(f"{C}/oof_final_gk_gbdt.npy", Pg)
