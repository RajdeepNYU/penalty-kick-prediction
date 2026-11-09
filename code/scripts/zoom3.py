"""Richer pixel features for the camera-aware pixel model (foot_sweep.py): the body crops read left/right far better than the foot crop once the head knows the camera and the foot. Per clip, frames nearest 0, 0.08, 0.2, 0.4 s before the end, stabilised to the last frame and centred on that frame's own body point;
views body (1.4 h on the hips), upper (0.7 h on the shoulders), legs (0.9 h on the knees); goal on the right as foot_zoom.py; 224 px; frozen DINOv3 ViT-B/16: 14 x 14 patch tokens pooled to 7 x 7, plus the CLS token; crop and mirror.
-> cache/zoom3_emb.npy [clips, view, time, mirror, 50, 768] float16 (index 0 = CLS), cache/zoom3_meta.npz. usage: python zoom3.py"""
import sys, pickle, numpy as np, cv2
from common import *

IMG = 224; VIEWS = ("body", "upper", "legs"); SIDES = dict(body=1.4, upper=0.7, legs=0.9); TIMES = (0.0, 0.08, 0.2, 0.4)


def crops(c):
    try:
        from late_reader import calib, axes
        b = load_raw("ballspot", c); k = load_raw("kpts", c)
        if b is None or k is None: return c, None
        n = min(b["nF"], len(k["kp"])); P_ = np.einsum("fij,fkj->fki", b["M"][:n], np.concatenate([k["kp"][:n], np.ones((n, k["kp"].shape[1], 1), np.float32)], -1)); P_ = P_[..., :2] / P_[..., 2:3]; h = max(float(np.asarray(k["box"])[-1][3] - np.asarray(k["box"])[-1][1]), 20.0); ts = np.asarray(b["ts"], float)[:n]; ts = ts - ts[-1]
        ids = [int(np.abs(ts + t).argmin()) for t in TIMES]; cal = calib(c, b); flip = bool(axes(*cal)[1][0] < 0) if cal is not None else False; fr = dict(zip(sorted(set(ids)), read_frames(c, [b["F"][i] for i in sorted(set(ids))]))); out = np.zeros((len(VIEWS), len(TIMES), IMG, IMG, 3), np.uint8)
        for ti, i in enumerate(ids):
            Pm = np.where((k["sc"][i] >= 2.0)[:, None], P_[i], np.nan); hip = np.nanmean(Pm[[11, 12]], 0); sho = np.nanmean(Pm[[5, 6]], 0); kn = np.nanmean(Pm[[13, 14]], 0); bx = np.asarray(k["box"])[min(i, len(k["box"]) - 1)]; cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
            hip = hip if np.isfinite(hip).all() else np.array([cx, cy], float); sho = sho if np.isfinite(sho).all() else hip + np.array([0.0, -0.3 * h]); kn = kn if np.isfinite(kn).all() else hip + np.array([0.0, 0.25 * h]); cen = dict(body=hip, upper=sho, legs=kn)
            for vi, v in enumerate(VIEWS):
                z = IMG / (SIDES[v] * h); T0 = np.array([[z, 0, IMG / 2 - cen[v][0] * z], [0, z, IMG / 2 - cen[v][1] * z], [0, 0, 1.0]]); im = cv2.warpPerspective(fr[i], T0 @ np.asarray(b["M"][i], float), (IMG, IMG), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                out[vi, ti] = cv2.cvtColor(np.ascontiguousarray(im[:, ::-1] if flip else im), cv2.COLOR_BGR2RGB)
        return c, (flip, out)
    except Exception:
        return c, None


if __name__ == "__main__":
    import torch, timm
    from multiprocessing import Pool
    Wt = ROOT + r"\Q4_word_GPT\models\vit_base_patch16_dinov3.lvd1689m.safetensors"; m = timm.create_model("vit_base_patch16_dinov3", pretrained=True, num_classes=0, pretrained_cfg_overlay=dict(file=Wt)).cuda().eval().half()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda().half(); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda().half()

    @torch.no_grad()
    def emb(arr):
        x = torch.from_numpy(np.ascontiguousarray(np.concatenate([arr, arr[:, :, ::-1]]))).cuda().permute(0, 3, 1, 2).half() / 255; t = m.forward_features((x - mean) / std); cls = t[:, 0].float(); p = t[:, m.num_prefix_tokens:]; g = int(round(p.shape[1] ** 0.5))
        p = p.reshape(len(x), g, g, -1).permute(0, 3, 1, 2); gr = torch.nn.functional.adaptive_avg_pool2d(p.float(), 7).flatten(2).permute(0, 2, 1); z = torch.cat([cls[:, None], gr], 1).cpu().numpy().astype(np.float16); n = len(arr)
        return np.stack([z[:n], z[n:]], 1)

    nV, nT = len(VIEWS), len(TIMES); G = np.lib.format.open_memmap(f"{PC}/zoom3_emb.npy", mode="w+", dtype=np.float16, shape=(len(CLIPS), nV, nT, 2, 50, 768)); ok = np.zeros(len(CLIPS), bool); fl = np.zeros(len(CLIPS), bool); ix = {c: j for j, c in enumerate(CLIPS)}; ex = []
    with Pool(int(os.environ.get("NPOOL", "10"))) as p:
        for n, (c, r) in enumerate(p.imap(crops, CLIPS, chunksize=4)):
            if r is None: continue
            G[ix[c]] = emb(r[1].reshape(-1, IMG, IMG, 3)).reshape(nV, nT, 2, 50, 768); ok[ix[c]] = True; fl[ix[c]] = r[0]
            if len(ex) < 6 and c in set(gt["clip"]): ex.append(np.vstack([np.hstack(list(r[1][v])) for v in range(nV)]))
            if (n + 1) % 500 == 0: print(n + 1, flush=True)
    G.flush(); cv2.imwrite(f"{P}/logs/zoom3_example.jpg", cv2.cvtColor(np.vstack(ex[:2]), cv2.COLOR_RGB2BGR)); np.savez(f"{PC}/zoom3_meta.npz", clips=np.array(CLIPS), ok=ok, flip=fl, views=np.array(VIEWS), times=np.array(TIMES)); print("saved", ok.sum(), "of", len(CLIPS))
