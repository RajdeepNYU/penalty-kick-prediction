"""Does super-resolution of the kicker crop make RTMW's keypoints more precise? Compare native RTMW vs RTMW on a Real-ESRGAN x2 upscaled crop: confidence and temporal jitter (second difference in stabilised coordinates)."""
import torch, cv2, os, pickle, time, numpy as np, pandas as pd
from rtmlib.tools.pose_estimation import RTMPose
from spandrel import ModelLoader
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
sr = ModelLoader().load_from_file(ROOT + r"\Q4_work\models\RealESRGAN_x2plus.pth").cuda().eval(); print("SR model:", type(sr.model).__name__, "scale", sr.scale)
URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip"; pose = RTMPose(URL, model_input_size=(288, 384), to_openpose=False, backend="onnxruntime", device="cuda")
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); av = pd.read_pickle(f"{C}/audit_final.pkl"); av = av[av.part == "train"].reset_index(drop=True)
rng = np.random.RandomState(0); small = np.where(av.kh < av.kh.quantile(0.4))[0]; rest = np.where(av.kh >= av.kh.quantile(0.4))[0]; sel = np.r_[rng.choice(small, 70, replace=False), rng.choice(rest, 40, replace=False)]
LEG = [11, 12, 13, 14, 15, 16]; rows = []; t0 = time.time()
def jitter(P): return np.nanmean(np.abs(P[2:] - 2 * P[1:-1] + P[:-2]), axis=(0, 2))
for i in sel:
    c = gt["clip"][i]; b = pickle.load(open(f"{C}/ballspot/{c}.pkl", "rb")); k = pickle.load(open(f"{C}/kpts/{c}.pkl", "rb"))
    cap = cv2.VideoCapture(f"{D}/train/{c}"); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    idx = list(range(b["nF"] - 8, b["nF"])); KS = np.zeros((8, 17, 2)); SS = np.zeros((8, 17))
    for q, i_ in enumerate(idx):
        im = fr[b["F"][i_]]; bx = k["box"][i_]; h = bx[3] - bx[1]; w = bx[2] - bx[0]; x0, y0 = int(max(bx[0] - 0.3 * w, 0)), int(max(bx[1] - 0.2 * h, 0)); x1, y1 = int(min(bx[2] + 0.3 * w, im.shape[1])), int(min(bx[3] + 0.2 * h, im.shape[0]))
        crop = im[y0:y1, x0:x1]; t = torch.from_numpy(crop[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().cuda() / 255
        with torch.no_grad(): up = sr(t.half() if False else t)
        up = (up[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1].copy(); s = up.shape[0] / crop.shape[0]
        nb = [(bx[0] - x0) * s, (bx[1] - y0) * s, (bx[2] - x0) * s, (bx[3] - y0) * s]; kp, sc = pose(up, bboxes=[nb]); KS[q] = kp[0][:17] / s + np.array([x0, y0]); SS[q] = sc[0][:17]
    Kb = k["kp"][idx][:, :17]; Sb = k["sc"][idx][:, :17]; M = b["M"][idx]
    st = lambda K: (np.einsum("fij,fkj->fki", M, np.concatenate([K, np.ones(K.shape[:2] + (1,))], -1)))[..., :2] / np.einsum("fij,fkj->fki", M, np.concatenate([K, np.ones(K.shape[:2] + (1,))], -1))[..., 2:3]
    hh = k["box"][-1][3] - k["box"][-1][1]
    rows.append(dict(small=av.kh[i] < av.kh.quantile(0.4), conf_nat=np.median(Sb[:, LEG]), conf_sr=np.median(SS[:, LEG]), jit_nat=jitter(st(Kb))[LEG].mean() / hh, jit_sr=jitter(st(KS))[LEG].mean() / hh, shift=np.mean(np.hypot(*(KS - Kb)[:, LEG].transpose(2, 0, 1))) / hh))
r = pd.DataFrame(rows); print(f"clips {len(r)} ({time.time() - t0:.0f}s)")
for nm, m in (("all", r.small | ~r.small), ("small players", r.small), ("larger players", ~r.small)):
    s = r[m]; print(f"  {nm:15s} n={len(s):3d} | leg confidence native {s.conf_nat.median():.2f} -> SR {s.conf_sr.median():.2f} | jitter (body heights) native {s.jit_nat.median():.4f} -> SR {s.jit_sr.median():.4f} ({(s.jit_sr.median() / s.jit_nat.median() - 1) * 100:+.0f}%) | mean keypoint shift {s['shift'].median():.4f} body heights")
print("share of clips where SR jitter is lower:", (r.jit_sr < r.jit_nat).mean().round(3))
