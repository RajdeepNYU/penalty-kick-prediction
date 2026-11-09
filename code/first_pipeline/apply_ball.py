"""Write the fused ball location (ball_final.pkl) back into cache/ballspot/<clip>.pkl for the clips where the consensus was weak (originals backed up in cache/ballspot_bak)."""
import pickle, os, shutil, numpy as np
ROOT = r"C:\path\to\project-root"; C = ROOT + r"\Q4_work\cache"; os.makedirs(f"{C}/ballspot_bak", exist_ok=True)
fin = pickle.load(open(f"{C}/ball_final.pkl", "rb")); n = 0
for c, r in fin.items():
    if r["method"] == "consensus": continue
    fp = f"{C}/ballspot/{c}.pkl"; b = pickle.load(open(fp, "rb"))
    if b is None: continue
    if not os.path.exists(f"{C}/ballspot_bak/{c}.pkl"): shutil.copy(fp, f"{C}/ballspot_bak/{c}.pkl")
    b["ball_last"] = np.asarray(r["ball"], float); b["ball_f"] = np.array([(np.linalg.inv(b["M"][i]) @ np.r_[b["ball_last"], 1.0])[:2] / (np.linalg.inv(b["M"][i]) @ np.r_[b["ball_last"], 1.0])[2] for i in range(b["nF"])])
    b["ball_method"] = r["method"]; pickle.dump(b, open(fp, "wb")); n += 1
print("ballspot files updated with the fused ball:", n)
