"""Roll out the expert and write clean / chaos datasets (one row per timestep).

Rows store the policy's (expert's) intended action as the label, even when the
executed action was corrupted by a noise burst.
"""
import argparse
import os
import time
from multiprocessing import Pool
import numpy as np
import pandas as pd
from env import MurphyEnv
from expert import run_episode
from chaos import Perturbation, TYPES

CLEAN_SEED0 = 100_000
CHAOS_SEED0 = 200_000


def rollout(job):
    kind, ep = job
    env = MurphyEnv()
    if kind == "clean":
        seed, pert = CLEAN_SEED0 + ep, None
    elif kind == "clean_extra":   # control set: more clean data, disjoint seeds
        seed, pert = CLEAN_SEED0 + 50_000 + ep, None
    else:
        seed = CHAOS_SEED0 + ep
        ptype = TYPES[np.random.default_rng([seed, 99]).integers(len(TYPES))]
        pert = Perturbation(ptype, seed)
    res = run_episode(env, seed, perturb=pert, record=True)
    info = pert.log() if pert else {"type": "none", "onset": None, "magnitude": 0.0}
    onset = -1 if info["onset"] is None else int(info["onset"])
    return [dict(episode_id=ep, seed=seed, t=t, obs=o.tolist(), action=a.tolist(),
                 perturbation_type=info["type"], perturbation_onset=onset,
                 perturbation_magnitude=float(info["magnitude"]), in_recovery=bool(rec),
                 episode_success=bool(res["success"])) for (t, o, a, rec) in res["rows"]]


def stats(df, name):
    ep = df.groupby("episode_id").agg(success=("episode_success", "first"),
                                     ptype=("perturbation_type", "first"),
                                     onset=("perturbation_onset", "first"), steps=("t", "size"))
    print(f"[{name}] episodes {len(ep)}  rows {len(df)}  success {ep.success.mean():.1%}")
    kept = df[df.episode_success]
    print(f"[{name}] kept for training: {kept.episode_id.nunique()} episodes, {len(kept)} rows, "
          f"in_recovery rows {kept.in_recovery.sum()} ({kept.in_recovery.mean():.1%})")
    if name == "chaos":
        print(ep.groupby("ptype").agg(n=("success", "size"), success=("success", "mean"),
                                      fired=("onset", lambda x: (x >= 0).mean()),
                                      mean_steps=("steps", "mean")).round(3).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--out", default="data")
    p.add_argument("--no-extra", action="store_true", help="skip the clean_extra control set")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for kind in ["clean", "chaos"] + ([] if args.no_extra else ["clean_extra"]):
        t0 = time.time()
        with Pool(args.workers) as pool:
            eps = pool.map(rollout, [(kind, i) for i in range(args.episodes)], chunksize=20)
        df = pd.DataFrame([r for e in eps for r in e])
        path = os.path.join(args.out, f"{kind}.parquet")
        df.to_parquet(path, index=False)
        print(f"wrote {path} in {time.time() - t0:.0f}s")
        stats(df, kind)
