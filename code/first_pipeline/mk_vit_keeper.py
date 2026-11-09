p = r"C:\path\to\project-root\Q4_work"; s = open(f"{p}\\kpts_vit.py", encoding="utf-8").read()
def rep(a, b):
    global s; assert a in s, a[:70]; s = s.replace(a, b, 1)
rep('on the kicker boxes of every frame of F', 'on the KEEPER boxes (cache/kkeep) of every frame of F'); rep('Writes cache/kptsE/<clip>.pkl', 'Writes cache/<VIT_OUT>/<clip>.pkl')
rep('k = pickle.load(open(f"{C}/kpts/{c}.pkl", "rb"))', 'k = pickle.load(open(f"{C}/kkeep/{c}.pkl", "rb"))')
open(f"{p}\\kpts_vit_keeper.py", "w", encoding="utf-8").write(s); print("wrote kpts_vit_keeper.py")
