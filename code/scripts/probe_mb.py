"""Calibrate MotionBERT-Lite output (cache/mb3d) in its camera frame (x right, y down, z depth): z sign by two anatomical handedness tests
(forward = (L - R) x up; the nose is in front of the thorax, a flexed knee points forward), z scale by bone-length constancy, and whether the lifted trunk
follows the camera elevation (a vertical trunk seen from elevation theta has camera-frame direction (0, -cos t, -sin t))."""
import pickle, numpy as np
from common import *

CAM = pickle.load(open(f"{PC}/camera2.pkl", "rb")); clips = [c for c in CLIPS if os.path.exists(f"{PC}/mb3d/{c}.pkl") and c in CAM]
unit = lambda V: V / np.linalg.norm(V, axis=-1, keepdims=True)
H = {1: [[], []], -1: [[], []]}; bones, tilt = [], []
for c in clips:
    d = pickle.load(open(f"{PC}/mb3d/{c}.pkl", "rb"))
    if d is None: continue
    J = d["j3"].astype(float); conf = d["j2"][..., 2]
    for s in (1, -1):
        Q = J * np.array([1, 1, s]); up = Q[:, 8] - Q[:, 0]
        fs = np.cross(Q[:, 11] - Q[:, 14], up); fh = np.cross(Q[:, 4] - Q[:, 1], up)
        H[s][0] += list(np.sign(np.einsum("ij,ij->i", fs, Q[:, 9] - Q[:, 8])))
        for hp, kn, an in ((1, 2, 3), (4, 5, 6)):
            off = Q[:, kn] - (Q[:, hp] + Q[:, an]) / 2; L = np.linalg.norm(Q[:, hp] - Q[:, an], axis=1); m = np.linalg.norm(off, axis=1) > 0.08 * L
            H[s][1] += list(np.sign(np.einsum("ij,ij->i", fh[m], off[m])))
    bones.append(np.stack([J[:, a] - J[:, b] for a, b in ((1, 2), (2, 3), (4, 5), (5, 6), (0, 8), (11, 12), (14, 15), (11, 14), (1, 4))], 1))
    t = unit(np.median(J[:, 8] - J[:, 0], 0)); tilt.append([np.sin(np.radians(CAM[c]["theta"])), t[2], t[1]])
print("clips", len(clips))
for s in (1, -1): print(f"z sign {s:+d}: nose-forward {np.mean(np.array(H[s][0]) > 0):.3f} (n={len(H[s][0])}) | flexed-knee-forward {np.mean(np.array(H[s][1]) > 0):.3f} (n={len(H[s][1])})")
for k in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
    cv = np.array([(lambda L: L.std(0) / L.mean(0))(np.sqrt(B[..., 0] ** 2 + B[..., 1] ** 2 + (k * B[..., 2]) ** 2)) for B in bones])   # within-clip constancy
    print(f"z scale {k:.2f}: within-clip bone-length CV {np.round(np.median(cv, 0), 3)} mean {np.median(cv, 0).mean():.3f}")
T = np.array(tilt); A = np.c_[T[:, 0], np.ones(len(T))]; sl = np.linalg.lstsq(A, T[:, 1], rcond=None)[0]
print(f"trunk z vs sin(theta): slope {sl[0]:+.2f} intercept {sl[1]:+.3f} (expected slope -1 if the camera elevation is captured; corr {np.corrcoef(T[:, 0], T[:, 1])[0, 1]:+.2f}) | sin(theta) range {T[:, 0].min():.2f}..{T[:, 0].max():.2f}")
