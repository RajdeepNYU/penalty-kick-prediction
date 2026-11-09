"""Alignment variants of the body crop, because random jitter of the crop costs most of the signal (px_sweep5.py): the reading depends on where the limbs fall in the crop. Last frame, stabilised, goal on the right as zoom3.py, 224 px, saved as images; default = 1.4 x last box height on the hips.
body_hm = scale from the median box height over the last 13 frames; body_tor = scale from the torso (mid-shoulder to mid-hip x 3.2); body_mid = centre halfway between hips and ankles; body_cen = centre of the last box; body_ball = centre on the ball; body11 = 1.1 h; body18 = 1.8 h.
-> cache/crops8.npy [clips, view, 224, 224, 3] uint8, cache/crops8_meta.npz. usage: python crops8.py"""
import pickle, numpy as np, cv2
from common import *

IMG = 224; VIEWS = ("body_hm", "body_tor", "body_mid", "body_cen", "body_ball", "body11", "body18"); B3 = pickle.load(open(f"{PC}/ball3.pkl", "rb"))


def crops(c):
    try:
        from late_reader import calib, axes
        b = load_raw("ballspot", c); k = load_raw("kpts", c)
        if b is None or k is None: return c, None
        n = min(b["nF"], len(k["kp"])); i = n - 1; Pi = (np.asarray(b["M"][i], float) @ np.c_[k["kp"][i], np.ones(len(k["kp"][i]))].T).T; Pi = Pi[:, :2] / Pi[:, 2:3]; bxs = np.asarray(k["box"]); h = max(float(bxs[-1][3] - bxs[-1][1]), 20.0); hm = max(float(np.median(bxs[-13:, 3] - bxs[-13:, 1])), 20.0); cal = calib(c, b); flip = bool(axes(*cal)[1][0] < 0) if cal is not None else False; f = read_frames(c, [b["F"][i]])[0]
        Pm = np.where((k["sc"][i] >= 2.0)[:, None], Pi, np.nan); bx = bxs[min(i, len(bxs) - 1)]; cen = np.array([(bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2], float); hip = np.nanmean(Pm[[11, 12]], 0); hip = hip if np.isfinite(hip).all() else cen; sho = np.nanmean(Pm[[5, 6]], 0); ank = np.nanmean(Pm[[15, 16]], 0); ank = ank if np.isfinite(ank).all() else hip + np.array([0.0, 0.45 * h]); tor = 3.2 * float(np.hypot(*(sho - hip))) if np.isfinite(sho).all() else h; tor = float(np.clip(tor, 0.6 * h, 1.6 * h)); ball = np.asarray(B3[c]["ball"], float) if B3.get(c) and np.isfinite(B3[c]["ball"][0]) else ank; out = np.zeros((len(VIEWS), IMG, IMG, 3), np.uint8)
        spec = dict(body_hm=(hip, 1.4 * hm), body_tor=(hip, 1.4 * tor), body_mid=((hip + ank) / 2, 1.4 * h), body_cen=(cen, 1.4 * h), body_ball=(ball, 1.4 * h), body11=(hip, 1.1 * h), body18=(hip, 1.8 * h))
        for vi, v in enumerate(VIEWS):
            q, sd = spec[v]; z = IMG / sd; T0 = np.array([[z, 0, IMG / 2 - q[0] * z], [0, z, IMG / 2 - q[1] * z], [0, 0, 1.0]]); im = cv2.warpPerspective(f, T0 @ np.asarray(b["M"][i], float), (IMG, IMG), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE); out[vi] = cv2.cvtColor(np.ascontiguousarray(im[:, ::-1] if flip else im), cv2.COLOR_BGR2RGB)
        return c, (flip, out)
    except Exception:
        return c, None


if __name__ == "__main__":
    from multiprocessing import Pool
    C_ = np.lib.format.open_memmap(f"{PC}/crops8.npy", mode="w+", dtype=np.uint8, shape=(len(CLIPS), len(VIEWS), IMG, IMG, 3)); ok = np.zeros(len(CLIPS), bool); fl = np.zeros(len(CLIPS), bool); ix = {c: j for j, c in enumerate(CLIPS)}; ex = []
    with Pool(int(os.environ.get("NPOOL", "8"))) as p:
        for n, (c, r) in enumerate(p.imap(crops, CLIPS, chunksize=4)):
            if r is None: continue
            C_[ix[c]] = r[1]; ok[ix[c]] = True; fl[ix[c]] = r[0]
            if len(ex) < 3 and c in set(gt["clip"]): ex.append(np.hstack(list(r[1])))
            if (n + 1) % 500 == 0: print(n + 1, flush=True)
    C_.flush(); cv2.imwrite(f"{P}/logs/crops8_example.jpg", cv2.cvtColor(np.vstack(ex), cv2.COLOR_RGB2BGR)); np.savez(f"{PC}/crops8_meta.npz", clips=np.array(CLIPS), ok=ok, flip=fl, views=np.array(VIEWS)); print("saved", ok.sum(), "of", len(CLIPS))
