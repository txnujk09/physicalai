"""Closed-loop scripted expert. Re-decides its phase from the CURRENT observation
every step (no internal memory), so knocked/dropped cubes get re-grasped."""
import argparse
import numpy as np
from env import MurphyEnv, CUBE_H, MAX_D

SAFE_Z = 0.10      # travel height (hand z)
GRASP_Z = CUBE_H   # hand z when fingers straddle a resting cube
CARRY_SPEED = 0.4  # action cap while holding the cube
OPEN_OK = 0.07     # opening considered "open"
PHASES = ["approach", "descend", "grasp", "lift", "move", "lower", "release"]


def phase_and_action(o, gain=1.0):
    h, opening, c, tgt = o[0:3], o[3], o[4:7], o[11:14]
    xy_err_cube = np.linalg.norm(h[:2] - c[:2])
    near_cube = xy_err_cube < 0.015 and abs(h[2] - c[2]) < 0.015
    grasped = near_cube and 0.025 < opening < 0.043
    cube_at_target = np.linalg.norm(c[:2] - tgt[:2]) < 0.01

    if grasped:
        if cube_at_target or np.linalg.norm(h[:2] - tgt[:2]) < 0.008:
            if c[2] < CUBE_H + 0.006:
                phase, goal, grip = "release", h, -1.0
            else:
                phase, goal, grip = "lower", np.array([tgt[0], tgt[1], GRASP_Z]), 1.0
        elif h[2] < SAFE_Z - 0.01 and np.linalg.norm(h[:2] - tgt[:2]) > 0.03:
            phase, goal, grip = "lift", np.array([h[0], h[1], SAFE_Z]), 1.0
        else:
            phase, goal, grip = "move", np.array([tgt[0], tgt[1], SAFE_Z]), 1.0
    else:
        resting_on_target = cube_at_target and c[2] < CUBE_H + 0.005
        if resting_on_target:
            # done with the cube: back off upward with fingers open
            phase, goal, grip = "release", np.array([h[0], h[1], SAFE_Z]), -1.0
        elif xy_err_cube > 0.008:
            # not above the cube: rise to safe height if low, then fly over it open
            if h[2] < SAFE_Z - 0.03 and xy_err_cube > 0.02:
                goal = np.array([h[0], h[1], SAFE_Z])
            else:
                goal = np.array([c[0], c[1], max(h[2], c[2] + 0.06)])
            phase, grip = "approach", -1.0
        elif h[2] - c[2] < 0.004:
            phase, goal, grip = "grasp", np.array([c[0], c[1], c[2]]), 1.0
        elif opening < OPEN_OK:
            # above cube but fingers not open (e.g. missed grasp): open in place
            phase, goal, grip = "descend", np.array([c[0], c[1], h[2] + 0.005]), -1.0
        else:
            phase, goal, grip = "descend", np.array([c[0], c[1], max(c[2], GRASP_Z)]), -1.0
    a = np.clip(gain * (goal - h) / MAX_D, -1, 1)
    if grasped:   # carry gently so a slippery grasp survives the acceleration
        a = np.clip(a, -CARRY_SPEED, CARRY_SPEED)
    return phase, np.array([*a, grip], dtype=np.float32)


class Expert:
    def __call__(self, o):
        return phase_and_action(o)[1]


def run_episode(env, seed, policy=None, perturb=None, record=False):
    """Roll one episode. perturb: optional chaos.Perturbation. Returns dict."""
    policy = policy or Expert()
    o = env.reset(seed=seed)
    if perturb is not None:
        perturb.bind(env)
    rows, done, info = [], False, {"success": False, "t": 0}
    while not done:
        a = np.asarray(policy(o), dtype=np.float32)
        a_exec = perturb.pre_step(env, a) if perturb is not None else a
        if record:
            rec = perturb.in_recovery(env.t) if perturb is not None else False
            rows.append((env.t, o.copy(), a.copy(), rec))
        o, r, done, info = env.step(a_exec)
    return {"success": bool(info["success"]), "steps": info["t"], "rows": rows}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    env = MurphyEnv()
    succ, steps = 0, []
    for i in range(args.episodes):
        res = run_episode(env, args.seed + i)
        succ += res["success"]
        if res["success"]:
            steps.append(res["steps"])
        elif args.verbose:
            print("fail seed", args.seed + i)
    print(f"expert success {succ}/{args.episodes} = {succ / args.episodes:.1%}, "
          f"mean steps (success) {np.mean(steps) if steps else float('nan'):.1f}")
