import numpy as np, sys
sys.argv = ["x"]
from cv import *
a, oof_h, cls = cv(X, Xm, "gk", seeds=(0, 1, 2))      # HGB classes sorted: left, right, stands in the middle
y = gt["goalkeeper_direction"].values
oof_c = np.load(f"{C}/oof_gk_resnet34.a1_in1k_0.npy")  # CNN order: left, stands, right
oof_c = oof_c[:, [0, 2, 1]]                             # -> left, right, stands
ok = np.isfinite(oof_c[:, 0])
print("HGB geometry %.3f | CNN %.3f (n=%d)" % (a, (np.array(cls)[oof_c[ok].argmax(1)] == y[ok]).mean(), ok.sum()))
for w in (0, 0.2, 0.3, 0.4, 0.5, 0.7):
    p = (1 - w) * oof_h[ok] + w * oof_c[ok]
    print(f"blend w_cnn={w:.1f}: acc {(np.array(cls)[p.argmax(1)] == y[ok]).mean():.3f}")
