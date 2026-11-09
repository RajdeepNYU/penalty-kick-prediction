"""Chain 23 (GPU): full-mode keeper NN (all public clips -> private clips), 5 configs x seeds {0,1,2} and {3,4,5}
-> cache/prvP_*_kp*_keeperAF.npy and cache/pg_private_NNk.npy (left, right, stands; mean of the 10 runs). -> logs/pipeline23.log"""
import time, numpy as np
from concurrent.futures import ThreadPoolExecutor
from runner import P, sh as _sh
LOG = open(f"{P}/logs/pipeline23.log", "a", buffering=1); sh = lambda a, env=None: _sh(LOG, a, env)
cfg = [(12, 96), (15, 128), (20, 96), (20, 128), (30, 128)]; env = dict(TARGET="keeper", BASE="AF", SEQ="kp", STREAMS="", MODE="full")
jobs = [(e, h, sd) for sd in ("0,1,2", "3,4,5") for e, h in cfg]
with ThreadPoolExecutor(3) as ex: list(ex.map(lambda j: sh(["train_seqP.py", str(j[0]), str(j[1])], dict(env, SEEDS=j[2])), jobs))
Q = sum(np.load(f"{P}/cache/prvP_{e}_{h}_st_kp{'' if sd == '0,1,2' else '_sd345'}_keeperAF.npy") for e, h, sd in jobs) / len(jobs)
np.save(f"{P}/cache/pg_private_NNk.npy", Q[:, [0, 2, 1]]); LOG.write(f"pg_private_NNk counts (left / right / stands) {np.bincount(Q[:, [0, 2, 1]].argmax(1), minlength=3)}\n")
LOG.write(f"[{time.strftime('%H:%M:%S')}] pipeline23 done\n")
