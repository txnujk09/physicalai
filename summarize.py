"""Print a markdown results table from results/results.json (+ extra training seeds if present)."""
import json
import os
import numpy as np

COND = ["clean", "impulse", "slip", "target", "friction", "noise", "all perturbed"]
runs = [p for p in ["results/results.json", "results/seed1/results.json", "results/seed2/results.json"] if os.path.exists(p)]
R = [json.load(open(p)) for p in runs]
r0 = R[0]["results"]
print(f"Seed-0 policies, {R[0]['meta']['episodes_per_condition']} eval episodes per condition (95% Wilson CI):\n")
pols = [p for p in ["A", "Aplus", "B", "expert"] if p in r0]
print("| condition | " + " | ".join(pols) + " |")
print("|---" * (len(pols) + 1) + "|")
for c in COND:
    cells = [f"{r0[p][c]['success_rate']*100:.1f}% ({r0[p][c]['wilson95'][0]*100:.0f}-{r0[p][c]['wilson95'][1]*100:.0f})" for p in pols]
    print(f"| {c} | " + " | ".join(cells) + " |")
if len(R) > 1:
    print(f"\nMean ± std over {len(R)} training seeds (same data, same eval seeds):\n")
    pols = ["A", "Aplus", "B"]
    print("| condition | " + " | ".join(pols) + " |")
    print("|---" * (len(pols) + 1) + "|")
    for c in COND:
        cells = []
        for p in pols:
            v = np.array([r["results"][p][c]["success_rate"] * 100 for r in R])
            cells.append(f"{v.mean():.1f} ± {v.std():.1f}")
        print(f"| {c} | " + " | ".join(cells) + " |")
