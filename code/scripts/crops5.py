"""More crops around the legs, because the knee-centred crop of the last frame reads left/right best (pixel_sweep3.py, px_blend.py). All at the last frame unless named, stabilised, goal on the right as zoom3.py, 224 px, saved as images:
legs_p1, legs_p2 = same legs crop one and two frames earlier; knee06 = 0.6 h on the knees; legs12 = 1.2 h on the knees; shin06 = 0.6 h on the ankles; foot09 = 0.9 h on the plant foot; ball09 = 0.9 h on the ball (ball3).
-> cache/crops5.npy [clips, view, 224, 224, 3] uint8, cache/crops5_meta.npz. usage: python crops5.py"""
import pickle, numpy as np, cv2
from common import *

IMG = 224; VIEWS = ("legs_p1", "legs_p2", "knee06", "legs12", "shin06", "foot09", "ball09"); B3 = pickle.load(open(f"{PC}/ball3.pkl", "rb"))


def crops(c):
    try:
        import kinR
        from late_reader import calib, axes
        b = load_raw("ballspot", c); k = load_raw("kpts", c)
        if b is None or k is None: return c, None
        n = min(b["nF"], len(k["kp"])); P_ = np.einsum("fij,fkj->fki", b["M"][:n], np.concatenate([k["kp"][:n], np.ones((n, k["kp"].shape[1], 1), np.float32)], -1)); P_ = P_[..., :2] / P_[..., 2:3]; h = max(float(np.asarray(k["box"])[-1][3] - np.asarray(k["box"])[-1][1]), 20.0); cal = calib(c, b); flip = bool(axes(*cal)[1][0] < 0) if cal is not None else False; ids = sorted({n - 1, max(n - 2, 0), max(n - 3, 0)}); fr = dict(zip(ids, read_frames(c, [b["F"][i] for i in ids])))
        lg, _, _ = kinR.legs(P_, k["sc"][:n], n); pts = np.array([[lg["pl"][t][nm] for nm in ("ank", "toe", "heel")] for t in range(max(0, n - 4), n)], float); ft = np.nanmedian(pts.reshape(-1, 2), 0) if np.isfinite(pts).any() else None; out = np.zeros((len(VIEWS), IMG, IMG, 3), np.uint8)

        def knees(i):
            Pm = np.where((k["sc"][i] >= 2.0)[:, None], P_[i], np.nan); bx = np.asarray(k["box"])[min(i, len(k["box"]) - 1)]; hip = np.nanmean(Pm[[11, 12]], 0); hip = hip if np.isfinite(hip).all() else np.array([(bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2], float); kn = np.nanmean(Pm[[13, 14]], 0); kn = kn if np.isfinite(kn).all() else hip + np.array([0.0, 0.25 * h]); an = np.nanmean(Pm[[15, 16]], 0); an = an if np.isfinite(an).all() else kn + np.array([0.0, 0.25 * h])
            return kn, an

        kn0, an0 = knees(n - 1); ft = ft if ft is not None else an0; ball = np.asarray(B3[c]["ball"], float) if B3.get(c) and np.isfinite(B3[c]["ball"][0]) else ft
        spec = dict(legs_p1=(max(n - 2, 0), None, 0.9), legs_p2=(max(n - 3, 0), None, 0.9), knee06=(n - 1, kn0, 0.6), legs12=(n - 1, kn0, 1.2), shin06=(n - 1, an0, 0.6), foot09=(n - 1, ft, 0.9), ball09=(n - 1, ball, 0.9))
        for vi, v in enumerate(VIEWS):
            i, cen, sd = spec[v]; cen = knees(i)[0] if cen is None else cen; z = IMG / (sd * h); T0 = np.array([[z, 0, IMG / 2 - cen[0] * z], [0, z, IMG / 2 - cen[1] * z], [0, 0, 1.0]]); im = cv2.warpPerspective(fr[i], T0 @ np.asarray(b["M"][i], float), (IMG, IMG), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE); out[vi] = cv2.cvtColor(np.ascontiguousarray(im[:, ::-1] if flip else im), cv2.COLOR_BGR2RGB)
        return c, (flip, out)
    except Exception:
        return c, None


if __name__ == "__main__":
    from multiprocessing import Pool
    C_ = np.lib.format.open_memmap(f"{PC}/crops5.npy", mode="w+", dtype=np.uint8, shape=(len(CLIPS), len(VIEWS), IMG, IMG, 3)); ok = np.zeros(len(CLIPS), bool); fl = np.zeros(len(CLIPS), bool); ix = {c: j for j, c in enumerate(CLIPS)}; ex = []
    with Pool(int(os.environ.get("NPOOL", "8"))) as p:
        for n, (c, r) in enumerate(p.imap(crops, CLIPS, chunksize=4)):
            if r is None: continue
            C_[ix[c]] = r[1]; ok[ix[c]] = True; fl[ix[c]] = r[0]
            if len(ex) < 3 and c in set(gt["clip"]): ex.append(np.hstack(list(r[1])))
            if (n + 1) % 500 == 0: print(n + 1, flush=True)
    C_.flush(); cv2.imwrite(f"{P}/logs/crops5_example.jpg", cv2.cvtColor(np.vstack(ex), cv2.COLOR_RGB2BGR)); np.savez(f"{PC}/crops5_meta.npz", clips=np.array(CLIPS), ok=ok, flip=fl, views=np.array(VIEWS)); print("saved", ok.sum(), "of", len(CLIPS))
