import numpy as np, pandas as pd, sys, warnings
warnings.filterwarnings("ignore")
import cv_kin
from cv_kin import *
P = np.load(f"{C}/oof_kin_bag.npy"); cl = np.array(["left", "middle", "right"]); pred = cl[P.argmax(1)]; ok = pred == yk
print("overall kick acc %.3f | L/R-only %.3f" % (ok.mean(), (pred[yk != "middle"] == yk[yk != "middle"]).mean()))
nm = list(names); g = lambda n: X[:, nm.index(n)]
d = pd.DataFrame(dict(ok=ok, conf=P.max(1), dur=-g("meta2") if "meta2" in nm else np.nan, support=g("meta3"), fps=g("meta1"), travel_ratio=g("foot_travel2"), kpq=g("kpq0"), fy=g("x_goalframe1"), fx=np.abs(g("x_goalframe0"))))
def seg(col, q=4, lab=None):
    s = pd.qcut(d[col], q, duplicates="drop"); t = d.groupby(s, observed=True).agg(n=("ok", "size"), acc=("ok", "mean")).round(3); print(f"-- {col}"); print(t.to_string())
for c_ in ("conf", "dur", "support", "fps", "travel_ratio", "kpq", "fx"): seg(c_)
bytrue = pd.crosstab(yk, pred); print(bytrue)
print("share of wrong predictions that are left/right swaps:", round(((pred != yk) & (pred != "middle") & (yk != "middle")).sum() / (pred != yk).sum(), 3))
# is the keeper's (true) side informative about kick beyond the model? (diagnostic only, uses labels)
yg = gt["goalkeeper_direction"].values; same = (yg == yk)
print("accuracy when true keeper side == kick side: %.3f (n=%d) vs not: %.3f (n=%d)" % (ok[same].mean(), same.sum(), ok[~same].mean(), (~same).sum()))
