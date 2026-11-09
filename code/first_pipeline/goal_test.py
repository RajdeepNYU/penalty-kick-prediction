import torch, numpy as np, cv2, pandas as pd, pickle, sys, time
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
R = r"C:\path\to\project-root"; D = R + r"\Q4data"; C = R + r"\Q4_work\cache"
S = r"scratch"
name = "IDEA-Research/grounding-dino-tiny"
proc = AutoProcessor.from_pretrained(name); model = AutoModelForZeroShotObjectDetection.from_pretrained(name).cuda().eval()
clips = ["0112_public.mp4", "0201_public.mp4", "0207_public.mp4", "0090_public.mp4", "0214_public.mp4", "0360_public.mp4", "0394_public.mp4", "0001_public.mp4", "0500_public.mp4", "0800_public.mp4", "1100_public.mp4", "0300_public.mp4"]
tiles = []; t0 = time.time()
for c in clips:
    cap = cv2.VideoCapture(f"{D}/train/{c}"); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    im = fr[-1]; rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    inp = proc(images=rgb, text="a soccer goal with net.", return_tensors="pt").to("cuda")
    with torch.no_grad(): out = model(**inp)
    res = proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=0.25, text_threshold=0.2, target_sizes=[rgb.shape[:2]])[0]
    best = res["boxes"].cpu().numpy()[np.argsort(-res["scores"].cpu().numpy())][:3] if len(res["boxes"]) else []
    for k, b in enumerate(best): cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), [(0, 255, 0), (0, 200, 255), (255, 0, 0)][k], 3)
    cv2.putText(im, f"{c[:4]} n={len(res['boxes'])} top={float(res['scores'].max()) if len(res['boxes']) else 0:.2f}", (8, 28), 0, 0.9, (0, 255, 255), 2); tiles.append(cv2.resize(im, (640, 360)))
print("s/frame", round((time.time() - t0) / len(clips), 2))
cv2.imwrite(f"{S}/goal_test.jpg", np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, 12, 2)]))
