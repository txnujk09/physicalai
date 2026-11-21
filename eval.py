"""Evaluate BC policies (and the expert for reference) on held-out seeds.

Every policy sees exactly the same seeds, and for perturbed conditions the same
perturbation schedule (type, planned onset, magnitude, kick direction) because
the perturbation RNG is seeded from the episode seed only. Eval seeds (900000+)
are disjoint from all training seeds (100000-152000, 200000-202000).
"""
import argparse
import json
import os
import time
from multiprocessing import Pool
import numpy as np

EVAL_SEED0 = 900_000
CONDITIONS = ["clean", "impulse", "slip", "target", "friction", "noise"]
LABELS = {"A": "A: clean only", "Aplus": "A+: 2x clean (control)", "B": "B: clean + chaos", "expert": "Scripted expert"}


def worker(job):
    import torch
    torch.set_num_threads(1)
    from env import MurphyEnv
    from expert import run_episode, Expert
    from chaos import Perturbation
    from train import TorchPolicy
    pol_name, cond, seeds, model_dir = job
    policy = Expert() if pol_name == "expert" else TorchPolicy(os.path.join(model_dir, f"{pol_name}.pt"))
    env = MurphyEnv()
    out = []
    for s in seeds:
        pert = None if cond == "clean" else Perturbation(cond, s)
        res = run_episode(env, s, policy=policy, perturb=pert)
        out.append(int(res["success"]))
    return pol_name, cond, seeds, out


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def plot(results, policies, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"A": "#2a78d6", "Aplus": "#eb6834", "B": "#1baf7a", "expert": "#b5b3ac"}
    conds = CONDITIONS + ["all perturbed"]
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    w = 0.8 / len(policies)
    x = np.arange(len(conds))
    for i, p in enumerate(policies):
        vals = [results[p][c]["success_rate"] * 100 for c in conds]
        bars = ax.bar(x + (i - (len(policies) - 1) / 2) * w, vals, w * 0.92, color=colors[p],
                      label=LABELS[p], edgecolor="#fcfcfb", linewidth=1)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", va="bottom",
                    fontsize=7.5, color="#52514e")
    ax.set_xticks(x, conds, color="#0b0b0b")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Success rate (%)", color="#52514e")
    n = results[policies[0]]["clean"]["n"]
    ax.set_title(f"Murphy: behaviour-cloned policy success, {n} identical seeds per condition",
                 loc="left", color="#0b0b0b")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    ax.yaxis.grid(True, color="#e8e7e3", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=len(policies), loc="upper left", bbox_to_anchor=(0, -0.08), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--policies", nargs="+", default=["A", "Aplus", "B", "expert"])
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--models", default="models")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--out", default="results")
    p.add_argument("--conditions", nargs="+", default=CONDITIONS)
    args = p.parse_args()
    CONDITIONS = args.conditions
    os.makedirs(args.out, exist_ok=True)
    seeds = list(range(EVAL_SEED0, EVAL_SEED0 + args.episodes))
    chunks = [seeds[i:i + 25] for i in range(0, len(seeds), 25)]
    jobs = [(pol, c, ch, args.models) for pol in args.policies for c in CONDITIONS for ch in chunks]
    t0 = time.time()
    with Pool(args.workers) as pool:
        raw = pool.map(worker, jobs, chunksize=1)
    per_ep = {pol: {c: {} for c in CONDITIONS} for pol in args.policies}
    for pol, c, ss, out in raw:
        per_ep[pol][c].update(dict(zip(ss, out)))
    results = {}
    for pol in args.policies:
        results[pol] = {}
        allk = alln = 0
        for c in CONDITIONS:
            v = [per_ep[pol][c][s] for s in seeds]
            k, n = sum(v), len(v)
            results[pol][c] = {"success_rate": k / n, "successes": k, "n": n, "wilson95": wilson(k, n)}
            if c != "clean":
                allk, alln = allk + k, alln + n
        results[pol]["all perturbed"] = {"success_rate": allk / alln, "successes": allk, "n": alln,
                                         "wilson95": wilson(allk, alln)}
    paired = {}
    if "A" in args.policies and "B" in args.policies:
        for c in CONDITIONS:
            a = np.array([per_ep["A"][c][s] for s in seeds])
            b = np.array([per_ep["B"][c][s] for s in seeds])
            paired[c] = {"B_only": int(((b == 1) & (a == 0)).sum()), "A_only": int(((a == 1) & (b == 0)).sum())}
    meta = {"eval_seeds": [seeds[0], seeds[-1]], "episodes_per_condition": args.episodes,
            "conditions": CONDITIONS, "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"meta": meta, "results": results, "paired_A_vs_B": paired,
                   "per_episode": {p_: {c: [per_ep[p_][c][s] for s in seeds] for c in CONDITIONS}
                                   for p_ in args.policies}}, f, indent=1)
    if set(CONDITIONS) == set(["clean", "impulse", "slip", "target", "friction", "noise"]):
        plot(results, args.policies, os.path.join(args.out, "success_rates.png"))
    hdr = f"{'condition':15s}" + "".join(f"{p_:>16s}" for p_ in args.policies)
    print(hdr)
    for c in CONDITIONS + ["all perturbed"]:
        print(f"{c:15s}" + "".join(f"{results[p_][c]['success_rate']*100:15.1f}%" for p_ in args.policies))
    if paired:
        print("paired (same seed) B succeeds & A fails / A succeeds & B fails:",
              {c: (v["B_only"], v["A_only"]) for c, v in paired.items()})
    print(f"wrote {args.out}/results.json and {args.out}/success_rates.png in {meta['seconds']}s")
