"""Run a script of this folder as a subprocess and append its (tail of) output to a log."""
import os, sys, subprocess, time
P = os.path.dirname(os.path.abspath(__file__)); os.makedirs(f"{P}/logs", exist_ok=True)


def sh(log, args, env=None):
    t0 = time.time(); r = subprocess.run([sys.executable] + args, cwd=P, capture_output=True, text=True, encoding="utf-8", errors="replace", env={**os.environ, **(env or {})})
    out = "\n".join([l for l in (r.stdout + r.stderr).splitlines() if l.strip() and "half' is deprecated" not in l][-40:]).encode("ascii", "replace").decode()   # logs are cp1252 files
    log.write(f"[{time.strftime('%H:%M:%S')}] {' '.join(args)} {env or ''} ({time.time() - t0:.0f}s, rc {r.returncode})\n{out}\n"); log.flush(); return r.returncode
