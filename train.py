"""Behaviour cloning: MLP obs -> action, MSE, normalised inputs.

A  : clean only (2000 episodes)
B  : clean + chaos (2000 + ~1940 episodes)
A+ : control, clean + clean_extra (4000 clean episodes) -- same data volume as B
All policies get the same number of gradient steps and batch size.
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from env import OBS_DIM, ACT_DIM

DATASETS = {"A": ["clean"], "B": ["clean", "chaos"], "Aplus": ["clean", "clean_extra"]}
# leave-one-out: B_no_<type> = clean + chaos without that perturbation type
for _t in ["impulse", "slip", "target", "friction", "noise"]:
    DATASETS[f"B_no_{_t}"] = ["clean", "chaos"]


def excluded_type(name):
    return name[5:] if name.startswith("B_no_") else None


class MLP(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.register_buffer("mu", torch.zeros(OBS_DIM))
        self.register_buffer("sd", torch.ones(OBS_DIM))
        self.net = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, ACT_DIM), nn.Tanh())

    def forward(self, o):
        return self.net((o - self.mu) / self.sd)


class TorchPolicy:
    def __init__(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.model = MLP(ck["hidden"])
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def __call__(self, o):
        return self.model(torch.as_tensor(o, dtype=torch.float32)[None])[0].numpy()


def load(names, data_dir, exclude=None):
    dfs = []
    for i, n in enumerate(names):
        df = pd.read_parquet(os.path.join(data_dir, f"{n}.parquet"))
        df = df[df.episode_success]
        if exclude is not None:
            df = df[df.perturbation_type != exclude]
        df = df.assign(ep_key=df.episode_id + 100_000 * i)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    return df


def train(name, args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    df = load(DATASETS[name], args.data, excluded_type(name))
    eps = df.ep_key.unique()
    val_eps = set(rng.choice(eps, size=max(1, len(eps) // 20), replace=False))
    is_val = df.ep_key.isin(val_eps).values
    O = np.stack(df.obs.values).astype(np.float32)
    A = np.stack(df.action.values).astype(np.float32)
    Otr, Atr, Ova, Ava = map(torch.from_numpy, (O[~is_val], A[~is_val], O[is_val], A[is_val]))
    model = MLP(args.hidden)
    model.mu.copy_(Otr.mean(0))
    model.sd.copy_(Otr.std(0).clamp_min(1e-3))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    g = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    print(f"[{name}] datasets {DATASETS[name]}: {len(eps)} episodes, {len(Otr)} train rows, {len(Ova)} val rows")
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(Otr), (args.batch,), generator=g)
        loss = ((model(Otr[idx]) - Atr[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % (args.steps // 5) == 0 or step == args.steps:
            with torch.no_grad():
                vl = ((model(Ova) - Ava) ** 2).mean().item()
            print(f"[{name}] step {step:6d}  train mse {loss.item():.4f}  val mse {vl:.4f}  ({time.time() - t0:.0f}s)")
    os.makedirs(args.out, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hidden": args.hidden, "datasets": DATASETS[name],
                "steps": args.steps, "val_mse": vl}, os.path.join(args.out, f"{name}.pt"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--policies", nargs="+", default=["A", "B", "Aplus"])
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="models")
    args = p.parse_args()
    torch.set_num_threads(os.cpu_count())
    for name in args.policies:
        train(name, args)
