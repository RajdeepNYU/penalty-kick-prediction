"""ViTPose+ tokens of the rebuilt crops (fix_crops.py), written over copies of the three P4rkC embedding files: cache/ce_{view}_vpbase{tag}.npy, so views e_{view}_vpbase{tag}. check = embed saved crops and compare with the saved tokens. usage: python cap.py fix_emb.py K | D | check"""
import sys, shutil, numpy as np
from common import *
from vp_embed import embed

TAG = sys.argv[1]; VW = ("body_hm", "body_tor", "legs12"); ix = {c: j for j, c in enumerate(CLIPS)}
if TAG == "check":
    C8 = np.load(f"{PC}/crops8.npy", mmap_mode="r"); M = list(np.load(f"{PC}/crops8_meta.npz")["views"]); J = [5, 500, 1500]; G = embed(np.array(C8[J, M.index("body_hm")])); E = np.load(f"{PC}/ce_body_hm_vpbase.npy", mmap_mode="r")[J]; print("max token difference from the saved tokens", float(np.abs(G.astype(np.float32) - E.astype(np.float32)).max()), "| token scale", float(np.abs(E.astype(np.float32)).mean())); sys.exit()
Z = np.load(f"{PC}/cropsfix_{TAG}.npz"); J = np.array([ix[c] for c in Z["clips"]])
for i, v in enumerate(VW):
    fp = f"{PC}/ce_{v}_vpbase{TAG}.npy"; shutil.copyfile(f"{PC}/ce_{v}_vpbase.npy", fp); G = np.load(fp, mmap_mode="r+"); G[J] = embed(Z["crops"][:, i]); G.flush(); del G; print("patched", fp, len(J), "clips", flush=True)
