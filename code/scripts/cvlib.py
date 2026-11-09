"""Shared CV protocol (identical to Q4_work: StratifiedGroupKFold(5, shuffle, seeds 0-2) by source_id, mirror augmentation + mirror TTA, HGB / ExtraTrees)."""
import numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier as HGB, ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from common import gt

cl = np.array(["left", "middle", "right"]); SW = {"left": "right", "right": "left", "middle": "middle"}
yk = gt["kick_direction"].values; ym = np.array([SW[v] for v in yk]); grp = gt["source_id"].values
yg = gt["goalkeeper_direction"].replace({"stands in the middle": "middle"}).values; ygm = np.array([SW[v] for v in yg])
hg = lambda s: HGB(random_state=s, max_iter=250, learning_rate=0.04, max_depth=4, min_samples_leaf=25, l2_regularization=2.0, max_features=0.5)
def lgb(s):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30, feature_fraction=0.3, bagging_fraction=0.8, bagging_freq=1,
                          lambda_l2=5.0, extra_trees=True, random_state=s, verbose=-1, n_jobs=8)


et = lambda s: make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(500, min_samples_leaf=3, max_features=0.3, n_jobs=-1, random_state=s))


def oof(mk, Xa, Xb, y=yk, y_m=ym, seeds=(0, 1, 2)):
    P = np.zeros((len(y), 3))
    for s in seeds:
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(Xa, y, grp):
            m = mk(s).fit(np.r_[Xa[tr], Xb[tr]], np.r_[y[tr], y_m[tr]]); c_ = list(m.classes_); ix = [c_.index(SW[k]) for k in c_]
            p = (m.predict_proba(Xa[te]) + m.predict_proba(Xb[te])[:, ix]) / 2; P[te] += p[:, [c_.index(k) for k in cl]] / len(seeds)
    return P


def oof_aug(mk, Xa, Xb, aug, y=yk, y_m=ym, seeds=(0, 1, 2)):
    """as oof, plus extra training rows: aug = list of (Aa, Ab, ok) aligned with the public clips (ok = row exists); test rows unchanged"""
    P = np.zeros((len(y), 3))
    for s in seeds:
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(Xa, y, grp):
            XA = [Xa[tr], Xb[tr]]; YA = [y[tr], y_m[tr]]
            for Aa, Ab, ok in aug: t2 = tr[ok[tr]]; XA += [Aa[t2], Ab[t2]]; YA += [y[t2], y_m[t2]]
            m = mk(s).fit(np.concatenate(XA), np.concatenate(YA)); c_ = list(m.classes_); ix = [c_.index(SW[k]) for k in c_]
            p = (m.predict_proba(Xa[te]) + m.predict_proba(Xb[te])[:, ix]) / 2; P[te] += p[:, [c_.index(k) for k in cl]] / len(seeds)
    return P


def fit_predict(mk, Xa, Xb, Qa, Qb, y=yk, y_m=ym, seeds=(0, 1, 2)):
    P = np.zeros((len(Qa), 3))
    for s in seeds:
        m = mk(s).fit(np.r_[Xa, Xb], np.r_[y, y_m]); c_ = list(m.classes_); ix = [c_.index(SW[k]) for k in c_]
        p = (m.predict_proba(Qa) + m.predict_proba(Qb)[:, ix]) / 2; P += p[:, [c_.index(k) for k in cl]] / len(seeds)
    return P


acc = lambda P, y=yk: float((cl[P.argmax(1)] == y).mean())
lr_acc = lambda P, y=yk: float((cl[P.argmax(1)] == y)[y != "middle"].mean())


def stack(rows, clips, dim):
    nan = np.full(dim, np.nan, np.float32)
    return np.stack([rows[c][0] if c in rows else nan for c in clips]), np.stack([rows[c][1] if c in rows else nan for c in clips])
