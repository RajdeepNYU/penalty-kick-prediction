"""Shared paths and loaders for the Q4_word_Prime experiments. Reads the existing Q4_work caches (never writes there)."""
import os, pickle, cv2, numpy as np, pandas as pd

ROOT = os.environ.get("Q4_ROOT", r"C:\path\to\project-root")
D = ROOT + r"\Q4data"
W = os.environ.get("Q4_CACHE", ROOT + r"\Q4_work\cache")            # existing pipeline caches (read-only); the notebook points this at its own copy
P = os.environ.get("Q4P_DIR", ROOT + r"\Q4_word_Prime")
PC = os.environ.get("Q4P_CACHE", P + r"\cache")                     # new caches
os.makedirs(PC, exist_ok=True)

gt = pd.read_csv(f"{D}/train_ground_truth.csv")
pm = pd.read_csv(f"{D}/private_metadata.csv")
CLIPS = list(gt["clip"]) + list(pm["clip"])
SPLIT = {c: "train" for c in gt["clip"]}
SPLIT.update({c: "val" for c in pm["clip"]})
META = {r["clip"]: dict(foot=r["preferred_foot"], gender=r["gender"], fps=float(r["fps"])) for _, r in pd.concat([gt, pm]).iterrows()}


OVL = os.environ.get("Q4_OVL")                                     # optional overlay of per-clip W caches + tracks.pkl (kfix.py: re-picked kicker), read first


GRIDM = os.environ.get("Q4_GRID")                                  # optional common time base for all frame rates (timegrid.py)


def load_raw(sub, c):
    fp = f"{OVL}/{sub}/{c}.pkl" if OVL and os.path.exists(f"{OVL}/{sub}/{c}.pkl") else f"{W}/{sub}/{c}.pkl"
    return pickle.load(open(fp, "rb")) if os.path.exists(fp) else None


VIEWJ = os.environ.get("Q4_VIEW")                                  # optional synthetic camera view (viewaug.py)


def load(sub, c):
    r = load_raw(sub, c)
    if VIEWJ and sub == "ballspot" and r is not None:
        import viewaug
        r = viewaug.view_b(c, r)
    if GRIDM and sub == "ballspot" and r is not None:
        import timegrid
        return timegrid.grid_b(r)
    return r


def read_frames(c, idx=None):
    cap = cv2.VideoCapture(f"{D}/{SPLIT[c]}/{c}"); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    return fr if idx is None else [fr[i] for i in idx]


_tracks = None
def tracks():
    global _tracks
    if _tracks is None:
        _tracks = pickle.load(open(f"{OVL}/tracks.pkl" if OVL and os.path.exists(f"{OVL}/tracks.pkl") else f"{W}/tracks.pkl", "rb"))
        if VIEWJ:
            import viewaug
            _tracks = {c: viewaug.view_tr(c, t) for c, t in _tracks.items()}
    return _tracks


def warp(M, pts):
    pts = np.asarray(pts, float); h = np.c_[pts.reshape(-1, 2), np.ones(len(pts.reshape(-1, 2)))] @ M.T
    return (h[:, :2] / h[:, 2:3]).reshape(pts.shape)
