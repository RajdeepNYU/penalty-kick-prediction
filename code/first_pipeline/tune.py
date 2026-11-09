"""Capacity / overfitting sweep for the GBDT and linear models (source-grouped CV, 3 seeds, mirror aug + TTA)."""
import numpy as np, pickle, sys, warnings, time
warnings.filterwarnings("ignore")
sys.argv = ["x"]
from cv import X, Xm, gt, SW, C
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

grp = gt["source_id"].values
KP = list(np.r_[0:90]) + [261]; KPL = list(np.r_[0:155]) + [261]; ALL = list(range(268))
KEEP = list(np.r_[155:268])


def run(mk, cols, col, swap=True, seeds=(0, 1, 2)):
    y = gt[col].values; ym = np.array([SW[v] for v in y]); Xa, Xb = X[:, cols], Xm[:, cols]; accs = []; tr_accs = []
    for s in seeds:
        pred = np.empty(len(y), object)
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(Xa, y, grp):
            m = mk(s).fit(np.r_[Xa[tr], Xb[tr]], np.r_[y[tr], ym[tr]]); cls = list(m.classes_); ix = [cls.index(SW[k]) for k in cls]
            p = (m.predict_proba(Xa[te]) + m.predict_proba(Xb[te])[:, ix]) / 2; pred[te] = np.array(cls)[p.argmax(1)]
            if s == seeds[0] and te[0] == StratifiedGroupKFold(5, shuffle=True, random_state=s).split(Xa, y, grp).__next__()[1][0]:
                tr_accs.append((m.predict(Xa[tr]) == y[tr]).mean())
        accs.append((pred == y).mean())
    return np.mean(accs), np.std(accs), (tr_accs[0] if tr_accs else np.nan)


hg = lambda **kw: (lambda s: HGB(random_state=s, **{**dict(max_iter=250, learning_rate=0.04, max_depth=3, min_samples_leaf=25, l2_regularization=2.0, max_features=0.5), **kw}))
lr = lambda Cr: (lambda s: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=Cr, max_iter=2000)))
t0 = time.time()
print("=== KICK (kicker pose + foot, 91 feats unless noted); train acc is on one fold's training data ===")
for nm, mk, cols in [
    ("hgb d3 it250 (current)", hg(), KP), ("hgb d3 it100", hg(max_iter=100), KP), ("hgb d3 it500", hg(max_iter=500), KP),
    ("hgb d2 it250", hg(max_depth=2), KP), ("hgb d4 it250", hg(max_depth=4), KP), ("hgb d3 leaf60", hg(min_samples_leaf=60), KP),
    ("hgb d3 l2=10", hg(l2_regularization=10.0), KP), ("hgb d3 it250 lr.02", hg(learning_rate=0.02), KP),
    ("logreg C=0.01", lr(0.01), KP), ("logreg C=0.1", lr(0.1), KP),
    ("hgb d3 it250 +lines/vel/ball (155)", hg(), KPL), ("hgb d3 it250 all 268", hg(), ALL)]:
    a, sd, ta = run(mk, cols, "kick_direction"); print(f"{nm:38s} cv {a:.3f} (sd over seeds {sd:.3f})  train {ta:.3f}  gap {ta - a:+.3f}   [{time.time() - t0:.0f}s]", flush=True)
print("\n=== KEEPER (all feats unless noted) ===")
for nm, mk, cols in [
    ("hgb d3 it250 all (current)", hg(), ALL), ("hgb d3 it100 all", hg(max_iter=100), ALL), ("hgb d3 it500 all", hg(max_iter=500), ALL),
    ("hgb d2 it250 all", hg(max_depth=2), ALL), ("hgb d4 it250 all", hg(max_depth=4), ALL), ("hgb d3 leaf60 all", hg(min_samples_leaf=60), ALL),
    ("hgb d3 it250 keeper-only(113)", hg(), KEEP), ("hgb d3 it250 keeper-only+kicker-static", hg(), KEEP + list(range(90))),
    ("logreg C=0.01 all", lr(0.01), ALL), ("logreg C=0.1 all", lr(0.1), ALL)]:
    a, sd, ta = run(mk, cols, "goalkeeper_direction"); print(f"{nm:38s} cv {a:.3f} (sd {sd:.3f})  train {ta:.3f}  gap {ta - a:+.3f}   [{time.time() - t0:.0f}s]", flush=True)
