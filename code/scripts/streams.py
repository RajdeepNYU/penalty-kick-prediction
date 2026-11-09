"""Keypoint streams in the RTMW-133 layout, as the Q4_work seqfeat*.py scripts feed them: A = RTMW-x (kpts), B = RTMW3D 2D (kpts3d, scores x10),
E/F/G = ViTPose++ base/large/huge (kptsE/F/G), S = Sapiens-0.3B body + feet (kptsS). Returns dict(kp, sc, box) with box from the RTMW stream, or None.
With Q4_GRID set (timegrid.py) the keypoints come back stabilised and on the common time grid."""
import os, numpy as np
from common import load, load_raw, GRIDM, VIEWJ


def _stream_kb(c, S):
    k0 = load("kpts", c)
    if k0 is None: return None
    if S == "A": return dict(kp=k0["kp"], sc=k0["sc"], box=k0["box"])
    if S == "M":                                                    # fused: per-frame, per-keypoint median of A, B, E, F, G for the 17 body points (robust to one
        ks = [_stream_kb(c, s) for s in "ABEFG"]                    # stream's leg swap or miss); RTMW-x feet / face / hands, the feet shifted with the fused ankles
        if any(k is None for k in ks): return None
        K = np.stack([k["kp"] for k in ks]); Sc = np.stack([k["sc"] for k in ks]); kp = k0["kp"].copy(); sc = k0["sc"].copy()
        kp[:, :17] = np.median(K[:, :, :17], 0); sc[:, :17] = np.median(Sc[:, :, :17], 0)
        for ank, ft in ((15, slice(17, 20)), (16, slice(20, 23))): kp[:, ft] += (kp[:, ank] - k0["kp"][:, ank])[:, None]
        return dict(kp=kp, sc=sc, box=k0["box"])
    if S == "B":
        k3 = load("kpts3d", c)
        if k3 is None: return None
        nF = k3["kp2"].shape[0]; kp = np.zeros((nF, 133, 2), np.float32); sc = np.zeros((nF, 133), np.float32); kp[:, :23] = k3["kp2"]; sc[:, :23] = k3["sc"] * 10
        return dict(kp=kp, sc=sc, box=k0["box"])
    ks = load({"E": "kptsE", "F": "kptsF", "G": "kptsG", "S": "kptsS"}[S], c)          # S = Sapiens-0.3B (kpts_sap.py, overlay cache/sapW)
    return None if ks is None else dict(kp=ks["kp"], sc=ks["sc"], box=k0["box"])


ANK = os.environ.get("Q4_ANK")                                      # faults 1 / 2 (RULES_fix.md): per-frame leg consensus, threshold in kicker heights (e.g. 0.10)
ANK_S = os.environ.get("Q4_ANK_S", "ABEF")                          # streams corrected; ViTPose+ huge (G) is left as is (lr_check.py: the consensus lowers its foot agreement)
LEG, LSW = [11, 12, 13, 14, 15, 16], [12, 11, 14, 13, 16, 15]; FT, FSW = list(range(17, 23)), [20, 21, 22, 17, 18, 19]


def leg_fix(c, S, kb, thr, ks=None):
    """Stream S's kicker hips / knees / ankles checked against the other pose streams (A, B, E, F, G) frame by frame. The other streams are put on
    common left / right labels (of all assignments, the one with the smallest summed pairwise leg distance; ties and the global flip go to the fewest
    relabelled), their median is the consensus, used only where the others agree (all but at most one within thr of it, per joint; for the
    relabel, on the six-joint mean). S is relabelled on a frame (legs and RTMW feet swapped) only when its legs match the consensus swapped
    clearly better (mean over the six joints by more than thr kicker heights); otherwise the consensus takes S's orientation, so overlapping legs never
    flip. Then a joint more than thr from the consensus is replaced by it on that frame only (score = the others' median score, not lowered), and RTMW
    feet move with their ankle. Returns (kb, n relabelled frames, n replaced joints)."""
    ks = ks or {s: _stream_kb(c, s) for s in "ABEFG"}; ks[S] = kb; nm = [s for s in "ABEFG" if s != S and ks[s] is not None and len(ks[s]["kp"]) == len(kb["kp"])]
    if len(nm) < 2: return kb, 0, 0
    SWI = [LEG.index(j) for j in LSW]; Y = np.stack([ks[s]["kp"][:, LEG].astype(float) for s in nm]); Ysw = Y[:, :, SWI]
    scn = np.stack([ks[s]["sc"][:, LEG] / (10.0 if s == "B" else 1.0) for s in nm]); scs = scn[:, :, SWI]
    n, nF = len(nm), len(kb["kp"]); h = np.maximum(kb["box"][:, 3] - kb["box"][:, 1], 1.0)
    d_id = np.full((n, n, nF), np.nan); d_sw = np.full((n, n, nF), np.nan)
    for i in range(n):
        for j in range(n):
            d_id[i, j] = np.nanmean(np.linalg.norm(Y[i] - Y[j], axis=-1), -1); d_sw[i, j] = np.nanmean(np.linalg.norm(Y[i] - Ysw[j], axis=-1), -1)
    labs = np.array([[(m >> k) & 1 for k in range(n)] for m in range(2 ** n)])           # 1 = stream relabelled
    kp = kb["kp"].copy(); sc = kb["sc"].copy(); med = np.full((nF, 6, 2), np.nan); msc = np.full((nF, 6), np.nan); rl = []
    for t in range(nF):
        best = None
        for o in labs:
            sw = o[:, None] ^ o[None, :]; cost = np.nansum(np.triu(np.where(sw == 1, d_sw[:, :, t], d_id[:, :, t]), 1))
            key = (round(float(cost) / h[t], 3), int(o.sum()))
            if best is None or key < best[0]: best = (key, o)
        o = best[1]; Al = np.where(o[:, None, None] == 1, Ysw[:, t], Y[:, t]); C = np.nanmedian(Al, 0); Cs = np.nanmedian(np.where(o[:, None] == 1, scs[:, t], scn[:, t]), 0)
        dk = np.linalg.norm(Al - C, axis=-1) / h[t]; need = max(2, n - 1)                      # the others must agree: at most one dissenter
        jok = (dk < thr).sum(0) >= need; lok = (np.nanmean(dk, 1) < thr).sum() >= need
        C = np.where(jok[:, None], C, np.nan)                                     # joints without agreement are left alone
        di, ds = np.nanmean(np.linalg.norm(kp[t, LEG] - C, axis=-1)) / h[t], np.nanmean(np.linalg.norm(kp[t, LEG] - C[SWI], axis=-1)) / h[t]
        if not lok or not np.isfinite(di) or not np.isfinite(ds): di = ds = 0.0
        if ds < di - thr:                                                          # clearly labelled the other way: relabel S
            kp[t, LEG] = kp[t, LSW]; sc[t, LEG] = sc[t, LSW]; rl.append(t)
            if (kb["sc"][t, FT] > 0).any(): kp[t, FT] = kp[t, FSW]; sc[t, FT] = sc[t, FSW]
        elif ds < di: C, Cs = C[SWI], Cs[SWI]                                     # ambiguous: the consensus takes S's orientation
        med[t], msc[t] = C, Cs * (10.0 if S == "B" else 1.0)
    d = np.linalg.norm(kp[:, LEG] - med, axis=-1) / h[:, None]; rep = np.isfinite(d) & (d > thr); nrep = int(rep.sum())
    old = kp[:, LEG].copy()
    for t, q in zip(*np.where(rep)):
        kp[t, LEG[q]] = med[t, q]; sc[t, LEG[q]] = max(float(msc[t, q]), float(sc[t, LEG[q]]))
    for ank, ft in ((15, slice(17, 20)), (16, slice(20, 23))):                    # RTMW feet follow a replaced ankle
        q = LEG.index(ank); mv = rep[:, q] & (kb["sc"][:, ft] > 0).any(1)
        kp[mv, ft] += (kp[mv, ank] - old[mv, q])[:, None]
    return dict(kb, kp=kp, sc=sc), len(rl), nrep


_LFC = {}


def _legfix_cached(c, S, kb):
    """leg_fix from the cache written by legfix_cache.py (same arrays), else computed"""
    if S not in _LFC:
        from common import PC
        fp = f"{PC}/legfix_{ANK}_{S}.pkl"; _LFC[S] = __import__("pickle").load(open(fp, "rb")) if os.path.exists(fp) else {}
    z = _LFC[S].get(c)
    if z is None or len(z[0]) != len(kb["kp"]): return leg_fix(c, S, kb, float(ANK))[0]
    kp, sc = kb["kp"].copy(), kb["sc"].copy(); kp[:, 11:23], sc[:, 11:23] = z; return dict(kb, kp=kp, sc=sc)


def stream_kb(c, S):
    kb = _stream_kb(c, S)
    if kb is not None and ANK and S in ANK_S: kb = _legfix_cached(c, S, kb)
    if kb is not None and VIEWJ:
        import viewaug
        kb = viewaug.view_kb(c, kb)
    if kb is None or not GRIDM: return kb
    import timegrid
    return timegrid.grid_kb(load_raw("ballspot", c), kb)
