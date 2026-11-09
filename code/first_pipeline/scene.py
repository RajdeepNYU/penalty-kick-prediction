"""Stage S1: scene understanding. For every clip, on the anchor frame (idx[6], 0.25 s before the end) and the last frame (idx[12]):
 - goal box(es) from GroundingDINO-tiny ('a soccer goal with net.'), top-3 with scores
 - extra person candidates from YOLO11x run on an upscaled crop around the goal mouth (small keepers are missed by the full-frame pass)
Writes cache/scene/<clip>.pkl."""
import torch, numpy as np, cv2, pandas as pd, pickle, os, time
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from ultralytics import YOLO
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"; OUT = f"{C}/scene"; os.makedirs(OUT, exist_ok=True)
name = "IDEA-Research/grounding-dino-tiny"; proc = AutoProcessor.from_pretrained(name); gdino = AutoModelForZeroShotObjectDetection.from_pretrained(name).cuda().eval()
yolo = YOLO(ROOT + r"\Q1-Q3work\yolo11x.pt")
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv")
todo = [("train", c) for c in gt["clip"]] + [("val", c) for c in pm["clip"]]; t0 = time.time(); n = 0
import sys
if len(sys.argv) > 1 and sys.argv[1] == "rev": todo = todo[::-1]        # second worker: walks the list backwards


def torso_colour(hsv, b):
    out = []
    for x0, y0, x1, y1 in b.astype(int):
        ty0 = y0 + (y1 - y0) // 5; ty1 = y0 + (y1 - y0) * 3 // 5; tx0 = x0 + (x1 - x0) // 4; tx1 = x1 - (x1 - x0) // 4
        t = hsv[max(ty0, 0):max(ty1, ty0 + 1), max(tx0, 0):max(tx1, tx0 + 1)].reshape(-1, 3)
        if len(t) == 0: t = np.zeros((1, 3), np.uint8)
        hh = np.bincount((t[:, 0] // 23).clip(0, 7), weights=t[:, 1] / 255., minlength=8) / len(t); out.append(np.r_[hh, t[:, 1].mean() / 255, t[:, 2].mean() / 255])
    return np.array(out).reshape(-1, 10)


def goals(im):
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB); inp = proc(images=rgb, text="a soccer goal with net.", return_tensors="pt").to("cuda")
    with torch.no_grad(): out = gdino(**inp)
    r = proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=0.2, text_threshold=0.15, target_sizes=[rgb.shape[:2]])[0]
    b = r["boxes"].cpu().numpy(); s = r["scores"].cpu().numpy(); o = np.argsort(-s)[:3]
    return np.c_[b[o], s[o]] if len(b) else np.zeros((0, 5), np.float32)


def zoom_persons(im, g):
    H, W = im.shape[:2]; x0, y0, x1, y1 = g[:4]; w, h = x1 - x0, y1 - y0; px, py = 0.35 * w, 0.6 * h
    X0, Y0, X1, Y1 = int(max(x0 - px, 0)), int(max(y0 - py, 0)), int(min(x1 + px, W)), int(min(y1 + 0.9 * h, H)); crop = im[Y0:Y1, X0:X1]
    if crop.shape[0] < 20 or crop.shape[1] < 20: return np.zeros((0, 5), np.float32)
    s = 1280 / max(crop.shape[:2]); up = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    r = yolo.predict(up, imgsz=1280, conf=0.1, classes=[0], verbose=False, device=0)[0]; b = r.boxes.xyxy.cpu().numpy() / s; c = r.boxes.conf.cpu().numpy()
    b[:, [0, 2]] += X0; b[:, [1, 3]] += Y0; return np.c_[b, c]


for split, c in todo:
    fp = f"{OUT}/{c}.pkl"
    if os.path.exists(fp): continue
    d = pickle.load(open(f"{C}/det/{c}.pkl", "rb")); cap = cv2.VideoCapture(f"{D}/{split}/{c}"); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    ia, il = d["idx"][6], d["idx"][12]; ga, gl = goals(fr[ia]), goals(fr[il]); ex = zoom_persons(fr[ia], ga[0]) if len(ga) else np.zeros((0, 5), np.float32)
    excol = torso_colour(cv2.cvtColor(fr[ia], cv2.COLOR_BGR2HSV), ex[:, :4]) if len(ex) else np.zeros((0, 10))
    pickle.dump(dict(goal_a=ga, goal_l=gl, extra_a=ex, extra_col=excol), open(fp, "wb")); n += 1
    if n % 100 == 0: print(n, f"{(time.time() - t0) / n:.2f}s/clip", flush=True)
print("done", n)
