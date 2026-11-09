"""Stage MB: MotionBERT-Lite 3D kicker pose (cache/mb3d, lift_mb.py) rotated into the goal frame with the per-clip camera (camera2.pkl).
probe_mb.py: the output frame is right-handed as is (nose in front 98%, flexed knee forward 99.8%) and its z units match x, y (bone-length constancy), so
camera vector -> ground = p @ R_of(cam). Lengths in body heights (median root-to-head-top + leg length), root-relative. Mirror = reflect v AND swap the L/R
joint labels (a pure reflection would flip handedness and point every facing vector backwards). Kicking leg = more 2D ankle travel over the last 0.2 s.
H36M joints: 0 root, 1-3 R hip/knee/ankle, 4-6 L hip/knee/ankle, 7 spine, 8 thorax, 9 nose, 10 head top, 11-13 L shoulder/elbow/wrist, 14-16 R.
-> cache/kinMB.pkl rows {clip: (static, static_mirror)}, names, seq {clip: (channels [12, 29] at the last 12 F frames, channels_mirror)}"""
import pickle, warnings, numpy as np
from common import *
warnings.filterwarnings("ignore")

FLIP = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13]
LEG = {0: dict(hip=1, knee=2, ank=3, sho=14, elb=15, wri=16), 1: dict(hip=4, knee=5, ank=6, sho=11, elb=12, wri=13)}
CAMF = os.environ.get("CAMF", "camera2"); CAM = pickle.load(open(f"{PC}/{CAMF}.pkl", "rb"))  # CAMF=camera2_ball3 -> kinMB_b3.pkl
TILT = os.environ.get("MB_TILT", "1") == "1"; TAG = os.environ.get("MB_TAG", "")             # MB_TILT=0: yaw-only rotation (probe_mb: lifted trunks stay camera-upright)
OUTN = ("kinMB" if TILT else "kinMB0") + TAG + {"camera2_ball3": "_b3", "camera2_ball3b": "_b3b", "camera2_cam3": "_c3"}.get(CAMF, "")     # MB_TAG=2: lifted from the 2 s window (cache/mb3d2)


def R_of(cam):
    p, t = np.radians(cam["phi"]), (np.radians(cam["theta"]) if TILT else 0.0); x = np.array([np.cos(p), -np.sin(p), 0]); y = np.array([-np.sin(t) * np.sin(p), -np.sin(t) * np.cos(p), -np.cos(t)])
    return np.stack([x, y, np.cross(x, y)])


def unit(V):
    n = np.linalg.norm(V, axis=-1, keepdims=True); return V / np.where(n > 1e-6, n, np.nan)


def slope(t, y):
    m = np.isfinite(y)
    return float(np.polyfit(t[m], y[m], 1)[0]) if m.sum() >= 2 and np.ptp(t[m]) > 0 else np.nan


def build(cam, d, mirror, F):
    f_, names = [], []
    def add(name, vals):
        vals = np.atleast_1d(np.asarray(vals, np.float32)).ravel(); f_.extend(vals.tolist()); names.extend([f"{name}{i}" if len(vals) > 1 else name for i in range(len(vals))])
    G = d["j3"].astype(float) @ R_of(cam); j2 = d["j2"].astype(float); ts = d["ts"] - d["ts"][-1]; n = len(ts)
    if mirror: G[..., 1] *= -1; G, j2 = G[:, FLIP], j2[:, FLIP]
    h = np.median(np.linalg.norm(G[:, 10] - G[:, 0], axis=1)) + np.median(np.linalg.norm(G[:, [1, 4]] - G[:, [2, 5]], axis=2).mean(1) + np.linalg.norm(G[:, [2, 5]] - G[:, [3, 6]], axis=2).mean(1))
    G = (G - G[:, :1]) / max(h, 1e-6); up = G[:, 8] - G[:, 0]
    w = lambda s: ts > -s - 1e-6; w1, w2, w3 = w(0.1), w(0.2), w(0.3); early = (ts > -0.6) & (ts < -0.4)
    hang = lambda V: np.arctan2(V[..., 1], V[..., 0])
    unwrap = lambda a: np.where(np.isfinite(a), np.unwrap(np.nan_to_num(a)), np.nan)
    # ---- kicking leg by 2D ankle travel over the last 0.2 s
    trav = [np.nansum(np.linalg.norm(np.diff(j2[w2][:, LEG[i]["ank"], :2], axis=0), axis=1)) for i in range(2)]; ki = int(np.argmax(trav))
    kk, pl = {k: G[:, v] for k, v in LEG[ki].items()}, {k: G[:, v] for k, v in LEG[1 - ki].items()}
    add("mb_kick_is_left", float(ki == 1)); add("mb_trav_ratio", trav[ki] / max(trav[1 - ki], 1e-6))
    # ---- horizontal body axes (mod 180) and full facing directions (handedness from the lifter)
    for nm, (a_, b_) in (("hip", (4, 1)), ("sho", (11, 14))):
        a2 = 2 * hang(G[:, a_] - G[:, b_]); add(f"mb_{nm}ax_last", [np.cos(a2[-1]), np.sin(a2[-1])]); add(f"mb_{nm}ax_m1", [np.nanmean(np.cos(a2[w1])), np.nanmean(np.sin(a2[w1]))])
        add(f"mb_{nm}ax_angvel", slope(ts[w3], unwrap(a2[w3]) / 2))
        fa = hang(np.cross(G[:, a_] - G[:, b_], up)); add(f"mb_{nm}face_last", [np.cos(fa[-1]), np.sin(fa[-1])]); add(f"mb_{nm}face_m1", [np.nanmean(np.cos(fa[w1])), np.nanmean(np.sin(fa[w1]))])
        add(f"mb_{nm}face_early", [np.nanmean(np.cos(fa[early])), np.nanmean(np.sin(fa[early]))] if early.any() else [np.nan, np.nan]); add(f"mb_{nm}face_angvel", slope(ts[w3], unwrap(fa[w3])))
    fh, fs = hang(np.cross(G[:, 4] - G[:, 1], up)), hang(np.cross(G[:, 11] - G[:, 14], up)); tw = np.angle(np.exp(1j * (fs - fh))); add("mb_twist", [tw[-1], np.nanmean(tw[w1])])
    na = hang(G[:, 9] - G[:, 8]); add("mb_nose_dir", [np.cos(na[-1]), np.sin(na[-1])])
    # ---- 3D segment directions (unit u, v, z)
    segs = {"kkth": (kk["hip"], kk["knee"]), "kksh": (kk["knee"], kk["ank"]), "kkleg": (kk["hip"], kk["ank"]), "plsh": (pl["knee"], pl["ank"]), "plleg": (pl["hip"], pl["ank"]),
            "trunk": (G[:, 0], G[:, 8]), "head": (G[:, 8], G[:, 10])}
    for nm, (a_, b_) in segs.items():
        U = unit(b_ - a_); add(f"mb_{nm}_last", U[-1]); add(f"mb_{nm}_m1", np.nanmean(U[w1], 0))
    add("mb_kkplane", np.nanmean(unit(np.cross(kk["knee"] - kk["hip"], kk["ank"] - kk["knee"]))[w1], 0))
    # ---- positions and velocities (body heights, root-relative)
    add("mb_kkank_rel", kk["ank"][-1]); add("mb_kkank_vs_pl", (kk["ank"] - pl["ank"])[-1]); add("mb_plank_rel", pl["ank"][-1]); add("mb_kkknee_rel", kk["knee"][-1])
    for nm, V in (("kkank", kk["ank"]), ("kkknee", kk["knee"])):
        for wn, ww in (("v1", w1), ("v2", w2)):
            vel = [slope(ts[ww], V[ww, q]) for q in range(3)]; add(f"mb_{nm}_{wn}", vel + [np.arctan2(vel[1], vel[0]) if np.isfinite(vel[0]) else np.nan])
    add("mb_wri_kkside", kk["wri"][-1]); add("mb_wri_other", pl["wri"][-1]); add("mb_sho_over_hip", (G[:, 8] - G[:, 0])[-1])
    add("mb_valid", [np.mean(j2[w3][..., 2]), d["swap_legs"], d["swap_arms"], n])
    # ---- per-frame channels for the sequence network at the last SEQ_T F frames (right-aligned, as kin3R)
    a2h, a2s = 2 * hang(G[:, 4] - G[:, 1]), 2 * hang(G[:, 11] - G[:, 14])
    ch = [np.stack([np.cos(a2h), np.sin(a2h), np.cos(a2s), np.sin(a2s), np.cos(fh), np.sin(fh), np.cos(fs), np.sin(fs)], 1), kk["ank"], kk["knee"], pl["ank"],
          unit(G[:, 8] - G[:, 0]), unit(kk["ank"] - kk["knee"]), unit(kk["knee"] - kk["hip"]), unit(pl["ank"] - pl["knee"])]
    X = np.clip(np.nan_to_num(np.concatenate(ch, 1), nan=0.0), -5, 5).astype(np.float32); ix = np.asarray(F) - d["fidx"][0]; ix = ix[(ix >= 0) & (ix < n)][-SEQ_T:]
    seq = np.zeros((SEQ_T, X.shape[1]), np.float32); seq[SEQ_T - len(ix):] = X[ix]
    return np.array(f_, np.float32), names, seq


SEQ_T = 12


def one(c):
    fp = f"{PC}/mb3d{TAG}/{c}.pkl"; cam = CAM.get(c); b, k = load("ballspot", c), load("kpts", c)
    if not os.path.exists(fp) or cam is None or b is None or k is None: return c, None, None
    d = pickle.load(open(fp, "rb"))
    if d is None or len(d["ts"]) < 4: return c, None, None
    F = b["F"][-min(len(b["F"]), len(k["box"])):]; a, names, sa = build(cam, d, False, F); m, _, sm = build(cam, d, True, F); return c, ((a, m), (sa, sm)), names


if __name__ == "__main__":
    from multiprocessing import Pool
    with Pool(int(os.environ.get("NPOOL", "12"))) as p: res = p.map(one, CLIPS, chunksize=20)
    rows = {c: r[0] for c, r, _ in res if r is not None}; seq = {c: r[1] for c, r, _ in res if r is not None}; names = next(n for _, r, n in res if r is not None)
    pickle.dump(dict(rows=rows, names=names, seq=seq), open(f"{PC}/{OUTN}.tmp", "wb")); os.replace(f"{PC}/{OUTN}.tmp", f"{PC}/{OUTN}.pkl")
    X = np.stack([r[0] for r in rows.values()]); print(f"{OUTN}: built {len(rows)} skipped {len(CLIPS) - len(rows)} | dim {len(names)} | NaN frac {np.isnan(X).mean():.3f}")
