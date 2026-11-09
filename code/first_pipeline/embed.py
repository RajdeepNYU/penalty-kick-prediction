"""Stage 6: frozen DINOv3-B/16 embeddings (CLS + mean patch) of tracked kicker / keeper crops, original and h-flipped."""
import pickle, os, sys, time, cv2, numpy as np, torch, timm

ROOT = r"C:\path\to\project-root"
C = ROOT + r"\Q4_work\cache"
W = ROOT + r"\Q4_word_GPT\models\vit_base_patch16_dinov3.lvd1689m.safetensors"
KF, GF = [12, 10, 8, 6], [12, 8, 6]
m = timm.create_model("vit_base_patch16_dinov3", pretrained=True, num_classes=0,
                      pretrained_cfg_overlay=dict(file=W)).cuda().eval().half()
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda().half(); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda().half()
os.makedirs(f"{C}/emb", exist_ok=True)


@torch.no_grad()
def embed(jpgs):
    ims = []
    for jp in jpgs:
        im = cv2.imdecode(np.frombuffer(jp, np.uint8), 1)[:, :, ::-1] if jp else np.zeros((256, 256, 3), np.uint8)
        im = cv2.resize(im, (224, 224)); ims += [im, im[:, ::-1]]
    x = torch.from_numpy(np.ascontiguousarray(np.stack(ims))).cuda().permute(0, 3, 1, 2).half() / 255
    t = m.forward_features((x - mean) / std)          # [N, 1+reg+P, D]
    cls = t[:, 0]; pt = t[:, m.num_prefix_tokens:].mean(1)
    e = torch.cat([cls, pt], 1).float().cpu().numpy().astype(np.float16)
    return e.reshape(len(jpgs), 2, -1)


clips = sorted(f[:-4] for f in os.listdir(f"{C}/pose"))
t0 = time.time(); n = 0
for c in clips:
    fp = f"{C}/emb/{c}.npy.pkl"
    if os.path.exists(fp): continue
    p = pickle.load(open(f"{C}/pose/{c}.pkl", "rb"))
    if p is None or p["keep"] is None: pickle.dump(None, open(fp, "wb")); continue
    out = dict(kick=embed([p["kick"]["jpg"][j] for j in KF]), keep=embed([p["keep"]["jpg"][j] for j in GF]),
               kmiss=[p["kick"]["jpg"][j] is None for j in KF], gmiss=[p["keep"]["jpg"][j] is None for j in GF])
    pickle.dump(out, open(fp, "wb")); n += 1
    if n % 100 == 0: print(n, f"{(time.time() - t0) / n:.2f}s/clip", flush=True)
print("done", n)
