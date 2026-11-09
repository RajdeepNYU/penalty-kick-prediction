"""One time base for every clip. The F frames were picked per frame rate, so the same input slot means different times: the last 12 slots span 0.8 s at 25-30 fps and 0.45 s at
50-60 fps, the last 6 span 0.2 s vs 0.083 s, the last 4 span 0.12 s vs 0.05 s. With Q4_GRID=1 in the environment, common.load("ballspot") and streams.stream_kb return every clip on the
25 fps pattern GRID (seconds before the end): keypoints are stabilised to last-frame coordinates (M becomes the identity) and linearly interpolated between the two nearest F frames,
so all feature code runs unchanged. Grid times before the first F frame are dropped. Q4_GRID_SHIFT=0.2 ends the window that many seconds before the clip's last frame
(an early-cut view of the same clip)."""
import os, numpy as np

GRID = -np.array([1.0, 0.8, 0.6, 0.44, 0.36, 0.28, 0.24, 0.2, 0.16, 0.12, 0.08, 0.04, 0.0]); SHIFT = float(os.environ.get("Q4_GRID_SHIFT", "0")); SMOOTH = os.environ.get("Q4_GRID_SMOOTH")
# Q4_GRID_PAT=15: the slot times the pipeline gives a real 15 fps clip (020_private: every frame back to -7/15 s, then -0.6, -0.8, -1.0), to simulate 15 fps on faster clips.
# Q4_GRID_PAT=15to25: sample on those 15 fps times first, then interpolate onto GRID, as a 15 fps clip would reach the 25 fps grid.
PAT = os.environ.get("Q4_GRID_PAT", "25"); GRID15 = -np.array([1.0, 0.8, 0.6, 7 / 15, 6 / 15, 5 / 15, 4 / 15, 3 / 15, 2 / 15, 1 / 15, 0.0])


def _times(b):
    t0 = np.asarray(b["ts"], float); t = t0[-1] - SHIFT + (GRID15 if PAT == "15" else GRID)
    return t0, t[t >= t0[0] - 1e-6]


def _lerp(t0, t, P, S):
    i = np.clip(np.searchsorted(t0, t, side="right") - 1, 0, len(t0) - 2); w = np.clip((t - t0[i]) / np.maximum(t0[i + 1] - t0[i], 1e-6), 0, 1)
    return (1 - w)[:, None, None] * P[i] + w[:, None, None] * P[i + 1], (1 - w)[:, None] * S[i] + w[:, None] * S[i + 1]


def grid_b(b):
    t0, t = _times(b)
    if len(t) < 3: return None                                                     # a clip shorter than the shift has no early view
    return dict(b, F=np.round(np.interp(t, t0, np.asarray(b["F"], float))).astype(int), ts=t, M=np.tile(np.eye(3, dtype=np.float32), (len(t), 1, 1)), nF=len(t))


def grid_kb(b, kb):
    """b = the raw ballspot row, kb = dict(kp [nF, K, 2], sc [nF, K], box) on the F frames"""
    t0, t = _times(b); kp, sc = np.asarray(kb["kp"]), np.asarray(kb["sc"]); P = np.einsum("fij,fkj->fki", b["M"], np.concatenate([kp, np.ones(kp.shape[:2] + (1,), kp.dtype)], -1)); P = P[..., :2] / P[..., 2:3]
    if PAT == "15to25":
        t15 = t0[-1] - SHIFT + GRID15; t15 = t15[t15 >= t0[0] - 1e-6]; P15, S15 = _lerp(t0, t15, P, sc); Q, Sq = _lerp(t15, t, P15, S15)
        return dict(kb, kp=Q.astype(np.float32), sc=Sq.astype(np.float32))
    i = np.clip(np.searchsorted(t0, t, side="right") - 1, 0, len(t0) - 2); w = np.clip((t - t0[i]) / np.maximum(t0[i + 1] - t0[i], 1e-6), 0, 1)
    Q = (1 - w)[:, None, None] * P[i] + w[:, None, None] * P[i + 1]
    if SMOOTH: Q = (np.r_[Q[:1], Q[:-1]] + 2 * Q + np.r_[Q[1:], Q[-1:]]) / 4                # Q4_GRID_SMOOTH=1: 1-2-1 over grid neighbours, against per-frame keypoint jitter
    return dict(kb, kp=Q.astype(np.float32), sc=((1 - w)[:, None] * sc[i] + w[:, None] * sc[i + 1]).astype(np.float32))
