"""MotionBERT-Lite 3D lifting of the dense kicker 2D (cache/kptsD). Model: walterzhu/MotionBERT FT_MB_lite_MB_ft_h36m_global_lite (DSTformer, dim_feat 256, depth 5,
dim_rep 512, 8 heads, att_fuse; models/motionbert/best_epoch.bin, weights loaded with weights_only=True). Run with python -I (model code is loaded by explicit path).
2D prep: L/R label continuity fix per group (legs; arms + face) by nearest assignment to the previous frame, then the global labelling that agrees with more raw frames;
COCO-17 -> H36M-17 (root = hip mid, thorax = shoulder mid, spine = their mid, nose, head top ~ ear mid + 0.5 (ear mid - thorax)); crop_scale to [-1, 1] over the sequence
and flip TTA, as infer_wild.py. Output in the model's camera frame (x right, y down, z depth; same units as x, y).
-> cache/mb3d/<clip>.pkl dict(fidx, ts, j2 [n, 17, 3] (normalized 2D + conf), j3 [n, 17, 3], swap_legs, swap_arms)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time, types, pickle, importlib.util, numpy as np, torch
from common import *

MB = P + r"\models\motionbert"; KIN = os.environ.get("KPTS", "kptsD"); OUT = f"{PC}/" + os.environ.get("MB_OUT", "mb3d"); os.makedirs(OUT, exist_ok=True)   # KPTS=kptsD2 MB_OUT=mb3d2: the 2 s window


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


for _n in ("lib", "lib.model"): sys.modules.setdefault(_n, types.ModuleType(_n))
_mod("lib.model.drop", MB + r"\drop.py"); DST = _mod("mb_dstformer", MB + r"\DSTformer.py"); UD = _mod("mb_utils_data", MB + r"\utils_data.py")

LEGS = [(11, 12), (13, 14), (15, 16)]; ARMS = [(5, 6), (7, 8), (9, 10), (1, 2), (3, 4)]


def fix_lr(kp, sc, pairs):
    kp, sc = kp.copy(), sc.copy(); L, R = [a for a, _ in pairs], [b for _, b in pairs]; sw = np.zeros(len(kp), bool)
    for j in range(1, len(kp)):
        pr, cu = kp[j - 1], kp[j]
        keep = np.linalg.norm(cu[L] - pr[L], axis=1).sum() + np.linalg.norm(cu[R] - pr[R], axis=1).sum()
        swap = np.linalg.norm(cu[R] - pr[L], axis=1).sum() + np.linalg.norm(cu[L] - pr[R], axis=1).sum()
        if swap < keep: kp[j, L + R] = kp[j, R + L]; sc[j, L + R] = sc[j, R + L]; sw[j] = True
    if sw.mean() > 0.5: kp[:, L + R] = kp[:, R + L]; sc[:, L + R] = sc[:, R + L]; sw = ~sw
    return kp, sc, float(sw.mean())


def coco2h36m(kp, sc):
    X = np.concatenate([kp, sc[..., None]], -1).astype(np.float32); Y = np.zeros((len(X), 17, 3), np.float32)
    mid = lambda a, b: np.concatenate([(X[:, a, :2] + X[:, b, :2]) / 2, np.minimum(X[:, a, 2:], X[:, b, 2:])], -1)
    Y[:, 0] = mid(11, 12); Y[:, 1], Y[:, 2], Y[:, 3], Y[:, 4], Y[:, 5], Y[:, 6] = X[:, 12], X[:, 14], X[:, 16], X[:, 11], X[:, 13], X[:, 15]
    Y[:, 8] = mid(5, 6); Y[:, 7, :2] = (Y[:, 0, :2] + Y[:, 8, :2]) / 2; Y[:, 7, 2] = np.minimum(Y[:, 0, 2], Y[:, 8, 2]); Y[:, 9] = X[:, 0]
    ear = mid(3, 4); Y[:, 10, :2] = ear[:, :2] + 0.5 * (ear[:, :2] - Y[:, 8, :2]); Y[:, 10, 2] = np.minimum(ear[:, 2], Y[:, 8, 2])
    Y[:, 11], Y[:, 12], Y[:, 13], Y[:, 14], Y[:, 15], Y[:, 16] = X[:, 5], X[:, 7], X[:, 9], X[:, 6], X[:, 8], X[:, 10]
    return Y


def load_model():
    m = DST.DSTformer(dim_in=3, dim_out=3, dim_feat=256, dim_rep=512, depth=5, num_heads=8, mlp_ratio=4, num_joints=17, maxlen=243, att_fuse=True)
    ck = torch.load(MB + r"\best_epoch.bin", map_location="cpu", weights_only=True)["model_pos"]
    m.load_state_dict({k.replace("module.", "", 1): v for k, v in ck.items()}, strict=True); return m.cuda().eval()


if __name__ == "__main__":
    model = load_model(); print("MotionBERT-Lite params", sum(p.numel() for p in model.parameters()))
    todo = [c for c in CLIPS if os.path.exists(f"{PC}/{KIN}/{c}.pkl") and not os.path.exists(f"{OUT}/{c}.pkl")]; t0 = time.time(); n = 0
    for c in todo:
        d = pickle.load(open(f"{PC}/{KIN}/{c}.pkl", "rb"))
        if d is None: pickle.dump(None, open(f"{OUT}/{c}.pkl", "wb")); continue
        kp, sc = d["kp"][-243:], d["sc"][-243:]; kp, sc, swl = fix_lr(kp, sc, LEGS); kp, sc, swa = fix_lr(kp, sc, ARMS)
        sc = np.where((sc > 0) & (kp != 0).any(-1), np.maximum(sc, 1e-3), 0.0)                       # frames without a crop stay conf 0 (ignored by crop_scale)
        j2 = UD.crop_scale(coco2h36m(kp, sc), [1, 1]).astype(np.float32); x = torch.from_numpy(j2)[None].cuda()
        with torch.no_grad(): y = (model(x) + torch.from_numpy(UD.flip_data(model(torch.from_numpy(UD.flip_data(j2))[None].cuda()).cpu().numpy())).cuda()) / 2
        pickle.dump(dict(fidx=d["fidx"][-243:], ts=d["ts"][-243:], j2=j2, j3=y[0].cpu().numpy(), swap_legs=swl, swap_arms=swa), open(f"{OUT}/{c}.pkl", "wb")); n += 1
    print("lifted", n, f"{time.time() - t0:.0f}s")
