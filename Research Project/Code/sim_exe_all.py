import os

scripts = [
    "ExVib_Closed.py",
    "ExVib_Coh_Open.py",
    "ExVib_Neg_Open.py",
    "ExVib_Pops_Open.py",
    "JCM_Closed.py",
    "JCM_Closed_eg.py",
    "JCM_Open.py",
    "JCM_Open_eg.py"
]

for script in scripts:
    os.system(f"python {script}")