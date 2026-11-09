import numpy as np, pickle, cv2, warnings
warnings.filterwarnings("ignore")
import cv_kin
from cv_kin import *
S = r"scratch"
M = pickle.load(open(f"{C}/oof_ens_lab2.pkl", "rb")); P = (M["hgb259"] + M["extratrees"]) / 2
cl = np.array(["left", "middle", "right"]); pred = cl[P.argmax(1)]; conf = P.max(1); wrong = (pred != yk) & (yk != "middle") & (pred != "middle")
print("confident (>0.8) predictions: n=%d acc=%.3f ; of the confident errors, left/right swaps: %d" % ((conf > 0.8).sum(), (pred[conf > 0.8] == yk[conf > 0.8]).mean(), (wrong & (conf > 0.8)).sum()))
# is the confident-error rate related to view geometry or fps?
fx = np.abs(X[:, list(names).index("x_goalframe0")]); fps = X[:, list(names).index("meta1")]
for nm, v in (("|goal dir x| (side-on=1)", fx), ("fps", fps)):
    q = np.quantile(v, [0, .33, .67, 1]); print(nm, [(f"{q[i]:.2f}-{q[i + 1]:.2f}", round(float(((pred == yk)[(conf > 0.8) & (v >= q[i]) & (v <= q[i + 1])]).mean()), 3)) for i in range(3)])
idx = np.where(wrong & (conf > 0.8))[0]; idx = idx[np.argsort(-conf[idx])][:12]
tiles = []
for i in idx:
    c = gt["clip"][i]; a = pickle.load(open(f"{C}/align/{c}.pkl", "rb"))
    row = np.hstack([cv2.resize(a["kick"][q][:, :, ::-1], (200, 200)) for q in (4, 2, 0)])
    cv2.rectangle(row, (0, 0), (600, 18), (0, 0, 0), -1); cv2.putText(row, f"{c[:4]} true={yk[i]} pred={pred[i]} conf={conf[i]:.2f} foot={gt['preferred_foot'][i]}", (3, 13), 0, 0.42, (0, 255, 255), 1); tiles.append(row)
cv2.imwrite(f"{S}/conf_errors.jpg", np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, 12, 2)])); print("ok (aligned crops: goal = up, kicker's left = screen-left; frames -0.15s, -0.06s, 0s)")
