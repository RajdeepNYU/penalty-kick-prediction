import nbformat, time
from nbclient import NotebookClient
R = r"C:\path\to\project-root\Q4_work"
nb = nbformat.read(f"{R}\\Q4_Penalty_Kick.ipynb", 4)
for c in nb.cells:   # executed copy: include the model-selection experiments
    if c.cell_type == "code" and "RUN_OPTIONAL_EVAL = False" in c.source:
        c.source = c.source.replace("RUN_OPTIONAL_EVAL = False", "RUN_OPTIONAL_EVAL = True")
t0 = time.time()
NotebookClient(nb, timeout=14400, kernel_name="python3", resources={"metadata": {"path": R}}).execute()
nbformat.write(nb, f"{R}\\Q4_Penalty_Kick_executed.ipynb"); print("executed OK in", round(time.time() - t0), "s")
