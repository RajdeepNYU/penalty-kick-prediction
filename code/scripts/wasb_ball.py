"""Item 2: second ball detector, WASB (Tarashima et al., BMVC 2023; nttcom/WASB-SBDT, MIT) soccer weights models/wasb/wasb_soccer_best.pth.tar: HRNet,
3 consecutive RGB frames in (letterboxed to 512 x 288, ImageNet normalisation), 3 heatmaps out; blobs = connected components of sigmoid > 0.5, heat-weighted
centroid and heat-sum score (the authors' tracknetv2 postprocessor). Model code is loaded by explicit path (run with python -I).
Run on triplets ending at each of the last 6 F frames (consecutive native frames); every blob of an F frame is warped to last-frame coords with that frame's M.
Clusters (radius 0.12 kicker heights): score = sum over frames of the best blob score x exp(-(d_foot / 0.3)^2), as ball2.
usage: python -I wasb_ball.py [N | clip ...]  -> cache/wasb/<clip>.pkl dict(blobs [k, 4] (F index, x, y, score), ball, score, support, d_foot)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time, pickle, importlib.util, numpy as np, cv2, torch, yaml
from common import *

SRC = P + r"\models\wasb_src\WASB-SBDT-main\src"; OUT = f"{PC}/wasb"; os.makedirs(OUT, exist_ok=True); IW, IH = 512, 288; NLAST = 6
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


class AD(dict):
    __getattr__ = lambda s, k: AD(s[k]) if isinstance(s[k], dict) else s[k]


def load_model():
    spec = importlib.util.spec_from_file_location("wasb_hrnet", SRC + r"\models\hrnet.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    cfg = AD(yaml.safe_load(open(SRC + r"\configs\model\wasb.yaml"))); net = m.HRNet(cfg)
    sd = torch.load(P + r"\models\wasb\wasb_soccer_best.pth.tar", map_location="cpu", weights_only=True)["model_state_dict"]
    net.load_state_dict({k.replace("module.", "", 1): v for k, v in sd.items()}, strict=True); return net.cuda().eval()


def letterbox(fr):
    h, w = fr.shape[:2]; k = min(IW / w, IH / h); A = np.array([[k, 0, (IW - k * w) / 2], [0, k, (IH - k * h) / 2]], np.float32)
    return cv2.warpAffine(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), A, (IW, IH), flags=cv2.INTER_LINEAR), A


def blobs(hm, thr=0.5):
    out = []
    if hm.max() <= thr: return out
    n, lab = cv2.connectedComponents((hm > thr).astype(np.uint8))
    for q in range(1, n):
        ys, xs = np.where(lab == q); w = hm[ys, xs]; out.append(((xs * w).sum() / w.sum(), (ys * w).sum() / w.sum(), float(w.sum())))
    return out


def foot_refs(b, c):
    pts = []
    for sub, ids in (("kpts", [15, 16, 17, 18, 19, 20, 21, 22]), ("kptsF", [15, 16]), ("kptsG", [15, 16])):
        k = load(sub, c)
        if k is None: continue
        for j in range(max(0, b["nF"] - 3), b["nF"]):
            sc = k["sc"][j, ids]; p = warp(b["M"][j], k["kp"][j, ids])[sc >= 1.5]
            if len(p): pts.append(p)
    return np.concatenate(pts) if pts else np.zeros((0, 2))


def one(net, c):
    b, k = load("ballspot", c), load("kpts", c)
    if b is None or k is None: return None
    F = list(b["F"]); nF = len(F); hL = max(float(k["box"][-1][3] - k["box"][-1][1]), 20.0); refs = foot_refs(b, c)
    want = sorted({f for i in range(max(0, nF - NLAST), nF) for f in (F[i] - 2, F[i] - 1, F[i]) if f >= 0}); fr = dict(zip(want, read_frames(c, want)))
    X = {}; A = None
    for f, im in fr.items():
        x, A = letterbox(im); X[f] = ((x.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)
    Ai = cv2.invertAffineTransform(A); det = {}
    trip = [(F[i] - 2, F[i] - 1, F[i]) for i in range(max(0, nF - NLAST), nF) if F[i] - 2 >= 0]
    with torch.no_grad():
        for t in trip:
            hm = torch.sigmoid(net(torch.from_numpy(np.concatenate([X[f] for f in t]))[None].cuda())[0])[0].cpu().numpy()
            for j, f in enumerate(t): det.setdefault(f, []).append(blobs(hm[j]))
    rows = []
    for i, f in enumerate(F):
        if f not in det: continue
        bl = max(det[f], key=lambda L: sum(s for *_, s in L))                     # the triplet giving this frame the most heat
        for x, y, s in bl: p = Ai @ np.array([x, y, 1.0]); q = warp(b["M"][i], p[None])[0]; rows.append((i, q[0], q[1], s))
    B = np.array(rows) if rows else np.zeros((0, 4)); rad = 0.12 * hL; best = None
    dfoot = lambda p: np.min(np.hypot(*(refs - p).T)) / hL if len(refs) else np.nan
    for p in B[:, 1:3]:
        near = np.hypot(B[:, 1] - p[0], B[:, 2] - p[1]) < rad; sup = sum(B[near & (B[:, 0] == fi), 3].max() for fi in np.unique(B[near, 0]))
        df = dfoot(p); sc = sup * (np.exp(-(df / 0.3) ** 2) if np.isfinite(df) else 1.0)
        if best is None or sc > best[0]:
            w = B[near, 3]; best = (sc, np.array([(B[near, 1] * w).sum() / w.sum(), (B[near, 2] * w).sum() / w.sum()]), int(len(np.unique(B[near, 0]))), df)
    r = dict(blobs=B.astype(np.float32), ball=None, score=0.0, support=0, d_foot=np.nan)
    if best is not None: r.update(score=float(best[0]), ball=best[1], support=best[2], d_foot=float(best[3]))
    return r


if __name__ == "__main__":
    net = load_model(); args = sys.argv[1:]
    todo = [a for a in args if a in SPLIT] or [c for c in CLIPS if not os.path.exists(f"{OUT}/{c}.pkl")][: int(args[0]) if args else None]; t0 = time.time()
    for n, c in enumerate(todo):
        r = one(net, c); pickle.dump(r, open(f"{OUT}/{c}.pkl", "wb"))
        if n % 200 == 0: print(n, c, f"{time.time() - t0:.0f}s", None if r is None else (r["ball"], round(r["score"], 2), r["support"]), flush=True)
    print("done", len(todo), f"{time.time() - t0:.0f}s")
