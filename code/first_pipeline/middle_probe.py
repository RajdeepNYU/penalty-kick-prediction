import numpy as np, sys, warnings
warnings.filterwarnings("ignore")
sys.argv = ["x"]
from cv import X, Xm, gt, C
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
y = gt["kick_direction"].values; grp = gt["source_id"].values; mid = (y == "middle").astype(int)
KP = list(np.r_[0:90]) + [261]; ALL = list(range(268))
P3 = np.load(f"{C}/oof_final_kick.npy")                 # current 3-class OOF (left, middle, right)
print("current 3-class model: mean P(middle) for true-middle %.3f vs others %.3f ; AUC %.3f" % (P3[mid == 1, 1].mean(), P3[mid == 0, 1].mean(), roc_auc_score(mid, P3[:, 1])))
for nm, cols in (("kicker pose+foot", KP), ("all feats", ALL)):
    oof = np.zeros(len(y))
    for s in (0, 1, 2):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, mid, grp):
            m = HGB(max_iter=200, learning_rate=0.04, max_depth=3, min_samples_leaf=25, l2_regularization=2.0, class_weight="balanced", random_state=s).fit(np.r_[X[tr][:, cols], Xm[tr][:, cols]], np.r_[mid[tr], mid[tr]])
            oof[te] += (m.predict_proba(X[te][:, cols])[:, 1] + m.predict_proba(Xm[te][:, cols])[:, 1]) / 2 / 3
    print(f"binary middle-vs-rest ({nm}): AUC {roc_auc_score(mid, oof):.3f}")
    if nm == "all feats":
        order = np.argsort(-oof)
        for k in (25, 50, 100):
            print(f"   top-{k} most 'middle'-like clips: precision {mid[order[:k]].mean():.2f} (base rate {mid.mean():.2f})")
# what would the score gain be if we predicted middle for the top-k with that precision?
Pk = P3.copy(); cl = np.array(["left", "middle", "right"]); base_acc = (cl[Pk.argmax(1)] == y).mean()
for k in (25, 50, 100):
    pred = cl[Pk.argmax(1)].copy(); pred[np.argsort(-oof)[:k]] = "middle"; print(f"force middle on top-{k}: kick acc {(pred == y).mean():.3f} (vs {base_acc:.3f})")
print("pose-block ablation for middle (what separates it?) - mean |z| of a few interpretable features, middle vs rest:")
names = {255: "keeper lateral offset (rel_v)", 256: "dist/h0", 259: "|fx| side-on", 261: "foot", 264: "clip duration"}
for i, n in names.items():
    a, b = X[mid == 1, i], X[mid == 0, i]; print(f"   {n:32s} mid {np.nanmean(a):+.3f} rest {np.nanmean(b):+.3f}  d={(np.nanmean(a) - np.nanmean(b)) / np.nanstd(X[:, i]):+.2f}")
