"""Stage 12: goal-aligned 16-frame clips over the final 0.6 s (resampled by timestamp), two crops: wide (body) and tight (feet/ball).
Writes cache/clips_wide.npy / clips_tight.npy  [2200,16,224,224,3] uint8 (order = train csv then private csv)."""
import cv2, pickle, time, numpy as np, pandas as pd
ROOT = r"C:\path\to\project-root"
D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
S, T, WIN = 224, 16, 0.6
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv")
clips = list(gt["clip"]) + list(pm["clip"]); split = ["train"] * len(gt) + ["val"] * len(pm)
tracks = pickle.load(open(f"{C}/tracks.pkl", "rb"))
N = len(clips)
mm = {v: np.lib.format.open_memmap(f"{C}/clips_{v}.npy", mode="w+", dtype=np.uint8, shape=(N, T, S, S, 3)) for v in ("wide", "tight")}
cen = lambda b: np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2]


def warp(im, cx, cy, side, ang):
    s = S / side; c, sn = np.cos(ang), np.sin(ang); R = np.array([[c, -sn], [sn, c]]) * s
    M = np.c_[R, np.array([S / 2, S / 2]) - R @ np.array([cx, cy])].astype(np.float32)
    return cv2.warpAffine(im, M, (S, S), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


t0 = time.time(); ok_flag = np.zeros(N, bool)
for n, (c, sp) in enumerate(zip(clips, split)):
    t = tracks[c]
    if t is None or t["keep"] is None: continue
    d = pickle.load(open(f"{C}/det/{c}.pkl", "rb")); kb = t["kick"]; v = np.isfinite(kb[:, 0])
    kc = cen(kb[v]); dts = np.array(d["ts"])[v]; o = np.argsort(dts); dts = dts[o]; kc = kc[o]
    hh = (kb[v, 3] - kb[v, 1])[o]; ww = (kb[v, 2] - kb[v, 0])[o]
    gc = cen(t["keep"]); vv = v & np.isfinite(gc[:, 0]); fv = (gc[vv] - cen(kb)[vv]).mean(0)
    ang = -np.pi / 2 - np.arctan2(fv[1], fv[0])
    cap = cv2.VideoCapture(f"{D}/{sp}/{c}"); fr, ts = [], []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f); ts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
    ts = np.array(ts); tt = np.linspace(ts[-1] - WIN, ts[-1], T)
    for q, tq in enumerate(tt):
        i = int(np.abs(ts - tq).argmin()); cx = np.interp(tq, dts, kc[:, 0]); cy = np.interp(tq, dts, kc[:, 1])
        h = np.interp(tq, dts, hh); w = np.interp(tq, dts, ww)
        mm["wide"][n, q] = warp(fr[i], cx, cy, 1.7 * max(h, 1.2 * w), ang)[:, :, ::-1]
        mm["tight"][n, q] = warp(fr[i], cx, cy + 0.2 * h, 0.95 * h, ang)[:, :, ::-1]
    ok_flag[n] = True
    if n % 200 == 0: print(n, f"{(time.time() - t0) / (n + 1):.2f}s/clip", flush=True)
for m_ in mm.values(): m_.flush()
np.save(f"{C}/clips_ok.npy", ok_flag); print("done", ok_flag.sum())
