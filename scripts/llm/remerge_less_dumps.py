import os, sys, torch
sys.path.insert(0, "/home/mila/a/ahmedm/gradmatch_selection/vendor/LESS")
from less.data_selection.collect_grad_reps import merge_and_normalize_info, merge_info
G = "/network/scratch/a/ahmedm/less/work/pythia410m/GRADS"
dirs = sorted(d for d in os.listdir(G) if os.path.isdir(os.path.join(G, d, "dim8192")))
for name in dirs:
    d = os.path.join(G, name, "dim8192")
    print("merging", name, flush=True)
    merge_and_normalize_info(d, prefix="grads")
    merge_info(d, prefix="grads")
print("DONE", len(dirs), "dirs")
