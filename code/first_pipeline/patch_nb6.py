import re
p = r"C:\path\to\project-root\Q4_work\build_nb6.py"; s = open(p, encoding="utf-8").read()


def rep(old, new, count=1):
    global s
    assert old in s, old[:70]
    s = s.replace(old, new, count)


# 1. adapt(): strip cross-module imports for the new scripts
rep('    s = s.replace("import kin2\\nfrom kin2 import legs_at, match2, slope\\n", "")\n',
    '    s = s.replace("import kin2\\nfrom kin2 import legs_at, match2, slope\\n", "")\n    s = s.replace("import kin2\\nfrom kin2 import legs_at, match2\\n", "").replace("CT = kin2.CT", "CT = 2.0").replace("import seqfeat\\n", "").replace("seqfeat.build(", "build(")\n')

# 2. results header
rep("| **v5 (this notebook)** | kick ensemble (GBDT+ExtraTrees); **camera-stabilised keeper features** + CNN | **0.652** | **0.896** | 0.722 | **0.724** | **0.72250** |",
    "| v5 | kick ensemble (GBDT+ExtraTrees); **camera-stabilised keeper features** + CNN | 0.652 | 0.896 | 0.722 | 0.724 | 0.72250 |\n| **v6 (this notebook)** | kick = 0.5·trees + 0.5·**temporal pose network** (two pose streams as measurement-noise augmentation) | **0.680** | **0.896** | 0.722 | **0.740** | **0.75916** |")
rep("CV and the public leaderboard agree within 0.002 for v5. The best public scores in the class are about 0.78, so there is still a gap that I could not close (Section 9).",
    "For v5, CV and the public leaderboard agreed within 0.002; v6 scored 0.019 *above* its CV estimate, which is within the noise of a 360-clip public split (about ±0.02), so I would not read it as more than a good sign. The best public scores in the class are about 0.78, so a small gap remains (Section 9).")
rep("* **ResNet34 (`resnet34.a1_in1k`, timm)**, ImageNet-pretrained, fine-tuned for keeper direction.",
    "* **RTMW3D-x** (`Soykaf/RTMW3D-x`, Hugging Face, ONNX), frozen: only its *2D* keypoints are used, as an independent second measurement of the same joints.\n* **ResNet34 (`resnet34.a1_in1k`, timm)**, ImageNet-pretrained, fine-tuned for keeper direction.")
rep("* Tried but **not in the final submission**: RTMW3D-x (`Soykaf/RTMW3D-x`, Hugging Face), ViTPose++ base", "* Tried but **not in the final submission**: RTMW3D's depth output, ViTPose++ base")

# 3. kpts3d cell after kpts_keeper
rep('cells.append(code(adapt(src("kpts_keeper.py"))))\n',
    'cells.append(code(adapt(src("kpts_keeper.py"))))\ncells.append(md("`kpts3d.py` runs **RTMW3D-x** on the same kicker boxes. Its 2D keypoints are a second, partly independent measurement of the same joints (used in Section 7b as noise augmentation); its depth output was tested and did not help (Section 9)."))\ncells.append(code(adapt(src("kpts3d.py"))))\n')

# 4. sequence feature cells after align
seq_cells = r'''cells.append(code(adapt(src("align.py"))))
cells.append(md(r"""
## 7b. Stage 6b - temporal pose network (the second big kick gain)
The tree models see hand-picked summaries of the pose. `seqfeat.py` instead exports the **per-frame label-free channels** for the last 12 frames (positions of kicking/plant foot, knee, hip, hip-mid, shoulder-mid and nose in the ball-anchored goal frame, hip/shoulder/ear axes mod 180°, gravity tilt of six body segments, time offset, validity) and `train_seq.py` trains a small **1D temporal-convolution network** on them with exact mirror augmentation (original or mirrored sample, label swapped) and mirror test-time averaging.

Two things made it work well:
1. **Measurement-noise augmentation.** Each clip has two independent pose measurements (RTMW, and RTMW3D's 2D keypoints; `seqfeatB.py`). The network trains on both as separate samples with the same label and averages both at test time: 0.652 → 0.658 alone, and the ensemble with the trees 0.670 → 0.679.
2. **Short training.** 12-20 epochs beat 60-100 (0.667-0.671 vs 0.633-0.650): the network overfits quickly with 1,300 clips.
Source-grouped CV for kick: bag of 5 network configurations 0.672 alone; **trees + network (0.5/0.5) 0.680** (left/right-only accuracy 0.776), against 0.652 for the trees. Variants that did *not* help: feeding the 259 hand-built features into the network (0.640 alone, worse ensemble), GRU (0.632) and transformer (0.621) encoders, seed bagging inside a fold.
"""))
cells.append(code(adapt(src("seqfeat.py"))))
cells.append(code(adapt(src("seqfeatB.py"))))
'''
rep('cells.append(code(adapt(src("align.py"))))\n', seq_cells)

# optional CV cells: train_seq loop + final_eval6, after final_eval5
opt = r'''cells.append(code(wrap(adapt(src("final_eval5.py")), "RUN_OPTIONAL_EVAL")))
ts_lines = [l for l in src("train_seq.py").replace('ROOT = r"C:\\path\\to\\project-root"', "ROOT = PROJECT_ROOT").split("\n") if "oof_ens_lab2" not in l and not l.startswith("base = (M[")]
ts_src = "\n".join(ts_lines)
assert (chr(39) * 3) not in ts_src and "EP = int(sys.argv[1])" in ts_src
cv_seq = "import sys, os\nTRAIN_SEQ_SRC = r\'\'\'" + ts_src + "\'\'\'\nif RUN_OPTIONAL_EVAL:\n    for ep, h in [(12, 96), (15, 128), (20, 96), (20, 128), (30, 128)]:\n        sys.argv = ['train_seq.py', str(ep), str(h)]; os.environ.update(KIN='0', SPF='1', ARCH='conv', AUGB='1')\n        exec(compile(TRAIN_SEQ_SRC, 'train_seq', 'exec'), {'__name__': 'seq_cv', 'PROJECT_ROOT': PROJECT_ROOT})"
cells.append(md("Optional: cross-validate the temporal network (5 configurations) and evaluate the final v6 kick/keeper configuration out of fold."))
cells.append(code(cv_seq))
fe6 = adapt(src("final_eval6.py")).replace('M = pickle.load(open(f"{C}/oof_ens_lab2.pkl", "rb")); trees = (M["hgb259"] + M["extratrees"]) / 2', "trees = (Ph + Pe) / 2")
fe6 = fe6.replace('KL = pickle.load(open(f"{C}/oof_keep_lab.pkl", "rb")); Pg = (KL["new_hgb"] + KL["new_et"] + KL["cnn"]) / 3; ', "")
cells.append(code(wrap(fe6, "RUN_OPTIONAL_EVAL")))
'''
rep('cells.append(code(wrap(adapt(src("final_eval5.py")), "RUN_OPTIONAL_EVAL")))\n', opt)

# 5. section 8: add final_seq + make_sub6
rep('cells.append(code(adapt(src("make_sub5.py"))))\n',
    'cells.append(code(adapt(src("make_sub5.py"))))\ncells.append(md(r"""\n**v6:** the temporal pose network bag (5 configurations x 3 seeds, trained on all 1,300 clips on both pose streams, original + mirrored, predictions averaged over streams and mirror) is blended 50/50 with the v5 tree probabilities for kick; the keeper is unchanged. This writes `submission_v6.csv`, the final submission.\n"""))\ncells.append(code(wrap(adapt(src("final_seq.py")), \x27FORCE_CNN or not os.path.exists(PROJECT_ROOT + r"\\\\Q4_work\\\\cache\\\\nn_private.npy")\x27)))\ncells.append(code(adapt(src("make_sub6.py"))))\n')
rep("The last cell writes `submission_v5.csv` and checks it against `submission_template.csv`.", "`make_sub5` writes `submission_v5.csv` (the previous best); the last cell writes `submission_v6.csv` and checks it against `submission_template.csv`.")

# 6. section 9 text
rep("* *Image models for kick:* frozen DINOv3 features 0.46-0.50, fine-tuned ResNet34 on kicker crops 0.49-0.50, MViTv2-S video model 0.51 (constant prediction). These used goal-aligned or kicker-only crops and may simply have been set up or trained badly; the pose-geometry route worked far better.",
    "* *Image models for kick:* frozen DINOv3 features 0.46-0.50, fine-tuned ResNet34 on kicker crops 0.49-0.50, MViTv2-S video model 0.51 (constant prediction), and an **end-to-end ResNet34 + temporal-conv model on homography-stabilised, ball-centred, goal-aligned 12-frame clips** (0.475 on fold 0 where the trees score 0.671; training loss only fell from 1.03 to 0.89). Three differently built image models all failed to learn beyond chance, while the same information expressed as pose geometry works, so with ~1,300 clips the cue (body lean) is not learnable from pixels with these recipes.")
rep("Public scores near 0.78 exist, so a stronger signal than the one I found is reachable; I do not know what it is.", "Public scores near 0.78 exist (v6 is at 0.759), so I may be missing some of what the best teams use; I do not know what it is.")
rep("(0.7238 vs 0.7225)", "(v5: 0.7238 vs 0.7225; v6: CV 0.740 vs public 0.759, a gap inside the ±0.02 noise of the 360-clip public split)")
rep("A left/right-only accuracy of 0.745 is therefore the ceiling-relevant number for kick.", "With the network bag the ensemble does predict some `middle` (40 clips out-of-fold, 40% correct against a 14% base rate; recall 0.09), but left/right-only accuracy (0.776) remains the number that matters for kick.")
rep("GPU training is not bit-for-bit deterministic, so a from-scratch run may change a few predictions.", "GPU training is not bit-for-bit deterministic, so a from-scratch run may change a few predictions: re-running the temporal-network bag changed 5 of 900 kick predictions relative to the file I submitted (submission_v6.csv; the rerun is saved as submission_v6_notebook_rerun.csv).")
rep("FORCE_CNN = False          # retrain the keeper CNN even if its predictions are cached", "FORCE_CNN = False          # retrain the keeper CNN and the temporal-network bag even if their predictions are cached")
open(p, "w", encoding="utf-8").write(s); print("patched")
