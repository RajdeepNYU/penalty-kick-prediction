"""Dense kicker 2D pose for 3D lifting: ViTPose++ large (usyd-community/vitpose-plus-large, COCO-17 head) on every native frame from F[0] to F[-1] (~1 s before the end).
Boxes: the kpts refined kicker boxes at the F frames, linearly interpolated per coordinate by frame index, enlarged x1.1. Each frame is cut to a padded square
around its box before the processor (same affine crop, far less memory); keypoints are shifted back to full-frame pixels.
-> cache/kptsD/<clip>.pkl dict(fidx [n], ts [n] s, kp [n, 17, 2] px, sc [n, 17], box [n, 4]). usage: python kpts_dense.py [N_CLIPS]"""
import sys, time, threading, queue, pickle, numpy as np, cv2, torch
from transformers import AutoProcessor, VitPoseForPoseEstimation
from common import *

OUT = f"{PC}/kptsD"; os.makedirs(OUT, exist_ok=True); NAME = "usyd-community/vitpose-plus-large"; BS, EXP, NTH = 64, 1.1, 4
proc = AutoProcessor.from_pretrained(NAME); model = VitPoseForPoseEstimation.from_pretrained(NAME).cuda().eval()


def crop(fr, bx):
    cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2; R = int(np.ceil(1.6 * max(bx[2] - bx[0], bx[3] - bx[1]))) + 4; H, Wd = fr.shape[:2]
    x0, y0 = int(cx) - R, int(cy) - R
    if not (0 <= cx < Wd and 0 <= cy < H): return None
    sub = fr[max(y0, 0):min(y0 + 2 * R, H), max(x0, 0):min(x0 + 2 * R, Wd)]
    sub = cv2.copyMakeBorder(sub, max(-y0, 0), max(y0 + 2 * R - H, 0), max(-x0, 0), max(x0 + 2 * R - Wd, 0), cv2.BORDER_CONSTANT, value=0)
    return cv2.cvtColor(sub, cv2.COLOR_BGR2RGB), np.array([[bx[0] - x0, bx[1] - y0, bx[2] - bx[0], bx[3] - bx[1]]], np.float32), np.array([x0, y0], np.float32)


def prepare(c):
    b, k = load("ballspot", c), load("kpts", c)
    if b is None or k is None: return c, None
    n = min(len(b["F"]), len(k["box"])); F = np.asarray(b["F"][-n:]); B = np.asarray(k["box"][-n:], float); fidx = np.arange(F[0], F[-1] + 1)
    box = np.stack([np.interp(fidx, F, B[:, i]) for i in range(4)], 1); ctr, half = (box[:, :2] + box[:, 2:]) / 2, (box[:, 2:] - box[:, :2]) / 2 * EXP; box = np.c_[ctr - half, ctr + half]
    cap = cv2.VideoCapture(f"{D}/{SPLIT[c]}/{c}"); j = 0; items = []
    while j <= F[-1]:
        ok, fr = cap.read()
        if not ok: break
        if j >= F[0]: items.append(crop(fr, box[j - F[0]]))
        j += 1
    items += [None] * (len(fidx) - len(items))
    return c, dict(fidx=fidx, ts=fidx / float(b["fps_native"]), box=box.astype(np.float32), items=items)


def producer(todo, q):
    while True:
        try: c = todo.get_nowait()
        except queue.Empty: break
        q.put(prepare(c))
    q.put(("__done__", None))


if __name__ == "__main__":
    clips = [c for c in CLIPS if not os.path.exists(f"{OUT}/{c}.pkl")]
    if len(sys.argv) > 1: clips = clips[:int(sys.argv[1])]
    todo = queue.Queue(); [todo.put(c) for c in clips]; q = queue.Queue(maxsize=8)
    for _ in range(NTH): threading.Thread(target=producer, args=(todo, q), daemon=True).start()
    t0, n, nfr, done = time.time(), 0, 0, 0
    while done < NTH:
        c, d = q.get()
        if c == "__done__": done += 1; continue
        if d is None: pickle.dump(None, open(f"{OUT}/{c}.pkl", "wb")); continue
        m = len(d["fidx"]); KP = np.zeros((m, 17, 2), np.float32); SC = np.zeros((m, 17), np.float32); ok = [i for i, it in enumerate(d["items"]) if it is not None]
        for s in range(0, len(ok), BS):
            idx = ok[s:s + BS]; ims = [d["items"][i][0] for i in idx]; bxs = [d["items"][i][1] for i in idx]
            inp = proc(ims, boxes=bxs, return_tensors="pt").to("cuda")
            with torch.no_grad(): out = model(**inp, dataset_index=torch.zeros(len(idx), dtype=torch.long, device="cuda"))
            res = proc.post_process_pose_estimation(out, boxes=bxs)
            for i, r in zip(idx, res): KP[i] = r[0]["keypoints"].cpu().numpy() + d["items"][i][2]; SC[i] = r[0]["scores"].cpu().numpy()
        pickle.dump(dict(fidx=d["fidx"], ts=d["ts"], kp=KP, sc=SC, box=d["box"]), open(f"{OUT}/{c}.tmp", "wb")); os.replace(f"{OUT}/{c}.tmp", f"{OUT}/{c}.pkl")
        n += 1; nfr += m
        if n % 100 == 0 or n == len(clips): print(f"{n}/{len(clips)} clips, {nfr} frames, {(time.time() - t0) / n:.2f} s/clip", flush=True)
    print("done", n, f"{time.time() - t0:.0f}s")
