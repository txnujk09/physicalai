"""Demos.

  video : side-by-side MP4 of policy A vs policy B on the same seed + perturbation
  live  : interactive viewer running a policy; drag the cube with the mouse
          (double-click the cube to select it, then Ctrl + right-drag to push it,
          Ctrl + left-drag to lift/rotate). On macOS run with `mjpython`.
"""
import argparse
import json
import os
import sys
import time
import numpy as np

COND_TEXT = {"impulse": "cube knocked", "slip": "grasp forced open", "target": "target teleported",
             "friction": "friction dropped", "noise": "action noise burst", "clean": "no perturbation"}


def make_cam(mujoco):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 0.04]
    cam.distance, cam.elevation, cam.azimuth = 0.62, -30, 120
    return cam


def pick_seed(results_path, cond):
    """Pick the first eval seed where B succeeded and A failed (illustrative case)."""
    with open(results_path) as f:
        r = json.load(f)
    a, b = r["per_episode"]["A"][cond], r["per_episode"]["B"][cond]
    s0 = r["meta"]["eval_seeds"][0]
    for i, (x, y) in enumerate(zip(a, b)):
        if y == 1 and x == 0:
            return s0 + i, True
    return s0, False


def video(args):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    import imageio
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.load_default(size=17)
    small = ImageFont.load_default(size=14)
    from env import MurphyEnv
    from chaos import Perturbation
    from train import TorchPolicy

    seed = args.seed
    if seed is None:
        seed, found = pick_seed(args.results, args.perturb)
        print(f"seed {seed}: " + ("first eval seed where B succeeds and A fails (selected as an illustrative case)"
                                   if found else "no B-wins-A-fails seed found; using first eval seed"))
    H, W = 360, 480
    sides = []
    for name in ["A", "B"]:
        env = MurphyEnv()
        pol = TorchPolicy(os.path.join(args.models, f"{name}.pt"))
        pert = None if args.perturb == "clean" else Perturbation(args.perturb, seed)
        o = env.reset(seed=seed)
        r = mujoco.Renderer(env.model, H, W)
        cam = make_cam(mujoco)
        frames, done, info, fired_at = [], False, {"success": False}, None
        while not done:
            a = pol(o)
            if pert is not None:
                a = pert.pre_step(env, a)
                if pert.onset is not None and fired_at is None:
                    fired_at = pert.onset
            o, _, done, info = env.step(a)
            r.update_scene(env.data, cam)
            frames.append((r.render().copy(), env.t, fired_at))
        r.close()
        sides.append((name, frames, info["success"], env.t))
    n = max(len(s[1]) for s in sides)
    out = []
    for i in range(n):
        panels = []
        for name, frames, success, T in sides:
            img, t, fired = frames[min(i, len(frames) - 1)]
            im = Image.fromarray(img)
            d = ImageDraw.Draw(im)
            title = "A: trained on clean demos" if name == "A" else "B: clean + chaos recovery demos"
            d.rectangle([0, 0, W, 46], fill=(20, 20, 20))
            d.text((8, 4), title, fill=(255, 255, 255), font=font)
            status = f"step {t}"
            if fired is not None and t >= fired:
                status += f"   PERTURBATION @ {fired}: {COND_TEXT[args.perturb]}"
            d.text((8, 25), status, font=small, fill=(255, 210, 80) if fired is not None and t >= fired else (200, 200, 200))
            if i >= len(frames) - 1:
                txt, col = (f"SUCCESS at step {T}", (40, 200, 90)) if success else (f"FAILED (timeout {T})", (230, 60, 60))
                d.rectangle([0, H - 40, W, H], fill=(20, 20, 20))
                d.text((8, H - 30), txt, fill=col, font=font)
            panels.append(np.asarray(im))
        out.append(np.concatenate(panels, axis=1))
    out += [out[-1]] * 40   # hold final frame
    path = args.out or f"results/demo_{args.perturb}_seed{seed}.mp4"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    imageio.mimsave(path, out, fps=args.fps, macro_block_size=8)
    print(f"A success={sides[0][2]} ({sides[0][3]} steps), B success={sides[1][2]} ({sides[1][3]} steps)")
    print("wrote", path)


def live(args):
    import mujoco
    import mujoco.viewer
    from env import MurphyEnv
    from chaos import Perturbation, TYPES
    from train import TorchPolicy
    from expert import Expert
    env = MurphyEnv()
    pol = Expert() if args.policy == "expert" else TorchPolicy(os.path.join(args.models, f"{args.policy}.pt"))
    seed = args.seed if args.seed is not None else 0
    rng = np.random.default_rng(seed)

    def new_episode(s):
        o = env.reset(seed=s)
        pert = Perturbation(TYPES[rng.integers(len(TYPES))], s) if args.chaos else None
        return o, pert

    o, pert = new_episode(seed)
    print(f"live: policy {args.policy}. Double-click the cube, then Ctrl+right-drag to shove it. "
          "Close the window to quit.")
    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.cam.lookat[:] = [0, 0, 0.04]
        v.cam.distance, v.cam.elevation, v.cam.azimuth = 0.8, -35, 120
        success_at = None
        while v.is_running():
            t0 = time.time()
            a = pol(o)
            if pert is not None:
                a = pert.pre_step(env, a)
            o, _, done, info = env.step(a)   # step cap ignored in live mode, only success resets
            v.sync()                          # also applies mouse perturbation forces
            if info["success"] and success_at is None:
                success_at = time.time()
                print(f"seed {seed}: success at step {env.t}" + (f" (perturbation {pert.log()})" if pert else ""))
            if success_at is not None and time.time() - success_at > 1.5:
                seed += 1
                o, pert = new_episode(seed)
                success_at = None
            time.sleep(max(0.0, env.model.opt.timestep * 10 - (time.time() - t0)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["video", "live"])
    p.add_argument("--perturb", default="impulse", choices=list(COND_TEXT))
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--policy", default="B", help="live mode: A, B, Aplus or expert")
    p.add_argument("--chaos", action="store_true", help="live mode: inject a random perturbation each episode")
    p.add_argument("--models", default="models")
    p.add_argument("--results", default="results/results.json")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    try:
        video(args) if args.mode == "video" else live(args)
    except KeyboardInterrupt:
        pass
