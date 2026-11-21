"""Perturbations injected at a random mid-episode step.

impulse  : cube velocity set to a random kick (magnitude = speed m/s)
slip     : gripper forced open for k steps once the cube is lifted (magnitude = k)
target   : target teleports to a new random spot (magnitude = jump distance m)
friction : cube + finger-pad friction drops for the rest of the episode (magnitude = mu)
noise    : Gaussian action noise burst for 15 steps (magnitude = sigma)
"""
import argparse
import numpy as np
from env import MurphyEnv, CUBE_H

TYPES = ["impulse", "slip", "target", "friction", "noise"]
ONSET_RANGE = (15, 60)      # expert finishes clean episodes in ~70 steps
NOISE_LEN = 15


class Perturbation:
    def __init__(self, ptype, seed):
        assert ptype in TYPES, ptype
        self.type = ptype
        self.rng = np.random.default_rng([seed, TYPES.index(ptype), 7])
        self.planned = int(self.rng.integers(*ONSET_RANGE))
        if ptype == "impulse":
            self.mag = float(self.rng.uniform(0.4, 1.0))
        elif ptype == "slip":
            self.mag = int(self.rng.integers(4, 10))
        elif ptype == "friction":
            self.mag = float(self.rng.uniform(0.08, 0.3))
        elif ptype == "noise":
            self.mag = float(self.rng.uniform(0.5, 1.5))
        else:
            self.mag = 0.0          # set when the target jumps
        self.onset = None           # actual step it fired
        self.left = 0

    def bind(self, env):
        self.onset, self.left = None, 0

    def _fire(self, env):
        self.onset = env.t
        if self.type == "impulse":
            th = self.rng.uniform(0, 2 * np.pi)
            v = self.mag * np.array([np.cos(th), np.sin(th), 0.3])
            env.kick_cube(v)
        elif self.type == "slip":
            self.left = self.mag
        elif self.type == "target":
            old = env.target.copy()
            while True:
                xy = self.rng.uniform(-0.15, 0.15, 2)
                if np.linalg.norm(xy - old[:2]) > 0.10 and np.linalg.norm(xy - env.cube_pos()[:2]) > 0.05:
                    break
            env.set_target(np.array([*xy, CUBE_H]))
            self.mag = float(np.linalg.norm(xy - old[:2]))
        elif self.type == "friction":
            env.set_grasp_friction(self.mag)
        elif self.type == "noise":
            self.left = NOISE_LEN

    def pre_step(self, env, a):
        """Called with the policy action before env.step; returns executed action."""
        if self.onset is None and env.t >= self.planned:
            # slip only makes sense once the cube is off the table
            if self.type != "slip" or env.cube_pos()[2] > CUBE_H + 0.015:
                self._fire(env)
        a = np.array(a, dtype=np.float32)
        if self.left > 0:
            if self.type == "slip":
                a[3] = -1.0
            elif self.type == "noise":
                a = np.clip(a + self.rng.normal(0, self.mag, 4), -1, 1).astype(np.float32)
            self.left -= 1
        return a

    def in_recovery(self, t):
        return self.onset is not None and t >= self.onset

    def log(self):
        return {"type": self.type, "onset": self.onset, "magnitude": self.mag}


if __name__ == "__main__":
    from expert import run_episode
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=5000)
    args = p.parse_args()
    env = MurphyEnv()
    print(f"expert under each perturbation, {args.episodes} episodes, seeds {args.seed}..")
    for pt in TYPES:
        succ, fired, steps, mags = 0, 0, [], []
        for i in range(args.episodes):
            pert = Perturbation(pt, args.seed + i)
            res = run_episode(env, args.seed + i, perturb=pert)
            succ += res["success"]
            fired += pert.onset is not None
            steps.append(res["steps"])
            mags.append(pert.mag)
        print(f"  {pt:9s} success {succ:3d}/{args.episodes}  fired {fired:3d}  "
              f"mean steps {np.mean(steps):5.1f}  magnitude {np.min(mags):.2f}-{np.max(mags):.2f}")
    print("example log:", Perturbation("impulse", 1).log())
