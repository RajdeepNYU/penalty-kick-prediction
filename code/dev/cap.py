"""CPU budget wrapper. History: 70% cap from 2026-11-01, 75% from 2026-11-04, lifted 2026-11-04 evening. 2026-11-05 01:58 the user asked to stop hogging the CPU
and use the GPU: jobs now run on logical CPUs 8-23 (16 of 24; CPUs 0-7 stay free for the user) at below-normal priority, inherited by every child process.
2026-11-05 17:00 the user: "you need to limit your usage to 75% gpu cpu max". CPUs 8-23 = 16 of 24 cores, so all jobs together use at most 67% of the CPU;
GPU scripts are held below 75% by gpucap.py. (A shared Windows job object with a 75% hard CPU-rate cap was tried for 20 minutes and dropped: processes already
in other Windows jobs could not join it, and every job ended without output while it was in place.)
2026-11-05 18:03 the user: "uncap gpu and cpu usage, you are permitted to use 100%": all 24 CPUs again; below-normal priority kept (it caps nothing, it only lets
the user's own apps go first). usage: python cap.py script.py args ..."""
import os, sys, subprocess, psutil

CPUS = list(range(psutil.cpu_count()))
p = psutil.Process(); p.cpu_affinity(CPUS); p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
sys.exit(subprocess.run([sys.executable] + sys.argv[1:], env=dict(os.environ, PYTHONUNBUFFERED="1")).returncode)
