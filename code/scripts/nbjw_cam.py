"""Item 1b: full perspective field calibration with No Bells Just Whistles (Gutierrez-Perez & Agudo 2024, mguti97/No-Bells-Just-Whistles; single-view
weights models/nbjw/SV_kp (57 field keypoints + background, HRNetV2-W48) and SV_lines (23 lines), loaded with weights_only=True). The authors'
inference.py steps, unchanged: RGB -> [0, 1] tensor resized to 540 x 960, max-pool keypoint extraction, thresholds 0.1486 (keypoints) / 0.3880 (lines),
complete_keypoints (line intersections), FramebyFrameCalib.heuristic_voting (cv2.calibrateCamera over modes / RANSAC levels, lowest reprojection error).
Model and utility code is loaded by explicit path (run with python -I). Run on the last F frame; the 3rd-last as a fallback.
World frame: pitch centre origin, x along the length (goals at x = +-52.5), z negative upwards (crossbar z = -2.44).
usage: python -I nbjw_cam.py [N | clip ...]  -> cache/nbjw/<clip>.pkl dict(P [3, 4] in original pixels, rep_err, mode, n_kp, frame, wh) or None"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time, pickle, importlib.util, numpy as np, cv2, torch, yaml
import torchvision.transforms as T
from common import *

NB = P + r"\models\nbjw_src\No-Bells-Just-Whistles-main"; OUT = f"{PC}/nbjw"; os.makedirs(OUT, exist_ok=True)


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


HR, HRL = _mod("nbjw_cls_hrnet", NB + r"\model\cls_hrnet.py"), _mod("nbjw_cls_hrnet_l", NB + r"\model\cls_hrnet_l.py")
UH, UC = _mod("nbjw_utils_heatmap", NB + r"\utils\utils_heatmap.py"), _mod("nbjw_utils_calib", NB + r"\utils\utils_calib.py")
RS = T.Resize((540, 960))


def frames_at(c, idx):
    """decode sequentially (as read_frames) but keep only the frames in idx"""
    cap = cv2.VideoCapture(f"{D}/{SPLIT[c]}/{c}"); out, i, want = {}, 0, set(idx)
    while i <= max(idx):
        ok, f = cap.read()
        if not ok: break
        if i in want: out[i] = f
        i += 1
    return out


def load_models():
    out = []
    for mod, cfgf, wf in ((HR, "hrnetv2_w48.yaml", "SV_kp"), (HRL, "hrnetv2_w48_l.yaml", "SV_lines")):
        m = mod.get_cls_net(yaml.safe_load(open(NB + "\\config\\" + cfgf))); m.load_state_dict(torch.load(P + "\\models\\nbjw\\" + wf, map_location="cpu", weights_only=True))
        out.append(m.cuda().eval())
    return out


def calib(models, fr):
    h, w = fr.shape[:2]; x = torch.from_numpy(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()[None] / 255
    x = (x if x.shape[-1] == 960 else RS(x)).cuda()
    with torch.no_grad(): hk, hl = models[0](x), models[1](x)
    kp = UH.coords_to_dict(UH.get_keypoints_from_heatmap_batch_maxpool(hk[:, :-1]), threshold=0.1486)
    ln = UH.coords_to_dict(UH.get_keypoints_from_heatmap_batch_maxpool_l(hl[:, :-1]), threshold=0.3880)
    fd = UH.complete_keypoints(kp, ln, w=960, h=540, normalize=True); cam = UC.FramebyFrameCalib(iwidth=w, iheight=h, denormalize=True); cam.update(fd[0])
    r = cam.heuristic_voting()
    if r is None: return None, len(fd[0])
    cp = r["cam_params"]; It = np.eye(4)[:-1]; It[:, -1] = -np.array(cp["position_meters"])
    Q = np.array([[cp["x_focal_length"], 0, cp["principal_point"][0]], [0, cp["y_focal_length"], cp["principal_point"][1]], [0, 0, 1]])
    return dict(P=Q @ (np.array(cp["rotation_matrix"]) @ It), rep_err=float(r["rep_err"]), mode=r["mode"], ransac=r["use_ransac"], n_kp=len(fd[0]), cam=cp, wh=(w, h)), len(fd[0])


if __name__ == "__main__":
    models = load_models(); args = sys.argv[1:]
    todo = [a for a in args if a in SPLIT] or [c for c in CLIPS if not os.path.exists(f"{OUT}/{c}.pkl")][: int(args[0]) if args else None]; t0 = time.time(); ok = 0
    for n, c in enumerate(todo):
        b = load("ballspot", c); r = None
        if b is not None:
            F = list(b["F"]); fr = frames_at(c, [F[i] for i in (len(F) - 1, len(F) - 3) if i >= 0])
            for i in (len(F) - 1, len(F) - 3):
                if i < 0 or F[i] not in fr: continue
                r, nk = calib(models, fr[F[i]])
                if r is not None: r["frame"] = i; break
        ok += r is not None; pickle.dump(r, open(f"{OUT}/{c}.pkl", "wb"))
        if n % 100 == 0: print(n, c, f"{time.time() - t0:.0f}s ok {ok}", None if r is None else (round(r["rep_err"], 2), r["mode"], r["n_kp"]), flush=True)
    print("done", len(todo), "calibrated", ok, f"{time.time() - t0:.0f}s")
