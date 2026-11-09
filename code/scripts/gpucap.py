"""GPU cap. History: 70% from 2026-11-01, 75% from 2026-11-04, lifted 2026-11-04 evening; 75% again 2026-11-05 17:00 (duty 0.7, memory fraction 0.75);
lifted 2026-11-05 18:03 (user: "uncap gpu and cpu usage, you are permitted to use 100%"): `with duty:` no longer sleeps and no memory fraction is set."""
import time, torch

SHARE = 1.0


class Duty:
    def __init__(self, share=1.0): self.r = (1 - share) / share

    def __enter__(self): self.t = time.perf_counter()

    def __exit__(self, *a):
        if self.r > 0: torch.cuda.synchronize(); time.sleep((time.perf_counter() - self.t) * self.r)


duty = Duty(SHARE)
