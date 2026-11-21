"""Murphy tabletop env: floating parallel-jaw gripper (mocap), one cube, one target.

Action (4,): [dx, dy, dz, grip] in [-1, 1].  dx..dz move the mocap hand by up to
MAX_D metres per step; grip is finger force: +1 squeezes with 4 N per finger, -1 opens.
Observation (14,): hand pos (3), gripper opening (1), cube pos (3), cube quat (4),
target pos (3).
Success: cube resting (speed < 5 cm/s, not touched by fingers) within 3 cm of
the target, held for SUCCESS_HOLD consecutive steps.
"""
import numpy as np
import mujoco

CUBE_H = 0.02          # cube half size
FINGER_MAX = 0.045     # per finger travel
MAX_D = 0.01           # metres per step per unit action
N_SUB = 10             # physics substeps per control step (dt=0.002 -> 50 Hz)
MAX_STEPS = 300
SUCCESS_DIST = 0.03
SUCCESS_HOLD = 5
WS_LO = np.array([-0.30, -0.30, 0.02])
WS_HI = np.array([0.30, 0.30, 0.25])
OBS_DIM = 14
ACT_DIM = 4

XML = """
<mujoco model="murphy">
  <option timestep="0.002" cone="elliptic" impratio="10" integrator="implicitfast"/>
  <visual><global offwidth="640" offheight="480"/><quality shadowsize="2048"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".85 .85 .8" rgb2=".75 .75 .7" width="256" height="256"/>
    <material name="grid" texture="grid" texrepeat="6 6"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1=".4 .5 .6" rgb2="0 0 0" width="32" height="32"/>
  </asset>
  <worldbody>
    <light pos="0 -0.4 1.2" dir="0 0.3 -1" directional="false" castshadow="true"/>
    <light pos="0 0 2" dir="0 0 -1" directional="true" castshadow="false" diffuse=".4 .4 .4"/>
    <geom name="table" type="plane" size="0.5 0.5 0.01" material="grid" friction="1 0.005 0.0001"/>
    <site name="target" type="cylinder" size="0.03 0.0015" pos="0.1 0.1 0.0015" rgba="0.1 0.9 0.2 0.5"/>
    <body name="hand_target" mocap="true" pos="0 0 0.15"/>
    <body name="hand" pos="0 0 0.15" gravcomp="1">
      <freejoint name="hand_j"/>
      <inertial pos="0 0 0.05" mass="0.3" diaginertia="0.001 0.001 0.001"/>
      <geom name="palm" type="box" size="0.012 0.06 0.008" pos="0 0 0.045" rgba=".25 .25 .3 1" contype="0" conaffinity="0"/>
      <geom name="stem" type="cylinder" size="0.01 0.05" pos="0 0 0.1" rgba=".25 .25 .3 1" contype="0" conaffinity="0"/>
      <body name="left_finger" gravcomp="1">
        <joint name="left_j" type="slide" axis="0 1 0" range="0 0.045" damping="25"/>
        <geom name="left_pad" type="box" size="0.012 0.006 0.018" pos="0 0.006 0" mass="0.02"
              friction="1.2 0.01 0.0001" condim="4" solref="0.01 1" rgba=".2 .4 .9 1"/>
        <geom name="left_link" type="box" size="0.006 0.004 0.014" pos="0 0.008 0.03" contype="0" conaffinity="0" mass="0.001" rgba=".2 .4 .9 1"/>
      </body>
      <body name="right_finger" gravcomp="1">
        <joint name="right_j" type="slide" axis="0 -1 0" range="0 0.045" damping="25"/>
        <geom name="right_pad" type="box" size="0.012 0.006 0.018" pos="0 -0.006 0" mass="0.02"
              friction="1.2 0.01 0.0001" condim="4" solref="0.01 1" rgba=".2 .4 .9 1"/>
        <geom name="right_link" type="box" size="0.006 0.004 0.014" pos="0 -0.008 0.03" contype="0" conaffinity="0" mass="0.001" rgba=".2 .4 .9 1"/>
      </body>
    </body>
    <body name="cube" pos="0 0 0.02">
      <freejoint name="cube_j"/>
      <geom name="cube" type="box" size="0.02 0.02 0.02" mass="0.05" friction="1.0 0.01 0.0001" condim="4" rgba=".9 .2 .15 1"/>
    </body>
  </worldbody>
  <contact><exclude body1="left_finger" body2="right_finger"/></contact>
  <equality><weld body1="hand_target" body2="hand" solref="0.01 1"/></equality>
  <actuator>
    <motor name="left_a" joint="left_j" ctrlrange="-1 1" gear="4"/>
    <motor name="right_a" joint="right_j" ctrlrange="-1 1" gear="4"/>
  </actuator>
</mujoco>
"""


class MurphyEnv:
    def __init__(self, seed=0):
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        m = self.model
        self.target_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "target")
        self.cube_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cube")
        self.pad_gids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("left_pad", "right_pad")}
        self.cube_qadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_j")]
        self.hand_qadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "hand_j")]
        self.hand_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.cube_vadr = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_j")]
        self.lj = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "left_j")]
        self.rj = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "right_j")]
        self.default_friction = m.geom_friction.copy()
        self.reset()

    # ------------------------------------------------------------------ api
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        m.geom_friction[:] = self.default_friction
        cube_xy = self.rng.uniform(-0.15, 0.15, 2)
        while True:
            tgt_xy = self.rng.uniform(-0.15, 0.15, 2)
            if np.linalg.norm(tgt_xy - cube_xy) > 0.10:
                break
        d.qpos[self.cube_qadr:self.cube_qadr + 3] = [*cube_xy, CUBE_H]
        d.qpos[self.cube_qadr + 3:self.cube_qadr + 7] = [1, 0, 0, 0]
        d.qpos[self.lj] = d.qpos[self.rj] = FINGER_MAX
        d.ctrl[:] = -1.0
        self.set_target(np.array([*tgt_xy, CUBE_H]))
        self.mocap = np.array([0.0, 0.0, 0.15])
        d.mocap_pos[0] = self.mocap
        d.qpos[self.hand_qadr:self.hand_qadr + 7] = [0, 0, 0.15, 1, 0, 0, 0]
        mujoco.mj_forward(m, d)
        self.t = 0
        self.hold = 0
        return self.obs()

    def step(self, action):
        self.apply_action(action)
        for _ in range(N_SUB):
            mujoco.mj_step(self.model, self.data)
        return self.post_step()

    def apply_action(self, action):
        """Write an action into mocap target + finger motors (no physics)."""
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        hand = self.hand_pos()
        self.mocap = np.clip(self.mocap + a[:3] * MAX_D, WS_LO, WS_HI)
        self.mocap = np.clip(self.mocap, hand - 0.02, hand + 0.02)   # anti wind-up
        self.data.mocap_pos[0] = self.mocap
        self.data.ctrl[:] = -a[3]   # +1 => closing force 4 N per finger

    def post_step(self):
        """Bookkeeping after N_SUB physics steps."""
        self.t += 1
        ok = self.success_now()
        self.hold = self.hold + 1 if ok else 0
        success = self.hold >= SUCCESS_HOLD
        done = success or self.t >= MAX_STEPS
        return self.obs(), float(success), done, {"success": success, "t": self.t}

    def obs(self):
        d = self.data
        return np.concatenate([
            self.hand_pos(),
            [d.qpos[self.lj] + d.qpos[self.rj]],
            d.qpos[self.cube_qadr:self.cube_qadr + 7],
            self.target,
        ]).astype(np.float32)

    # -------------------------------------------------------------- helpers
    def set_target(self, pos):
        self.target = np.array(pos, dtype=np.float64)
        self.model.site_pos[self.target_sid] = [pos[0], pos[1], 0.0015]

    def hand_pos(self):
        return self.data.qpos[self.hand_qadr:self.hand_qadr + 3].copy()

    def cube_pos(self):
        return self.data.qpos[self.cube_qadr:self.cube_qadr + 3].copy()

    def cube_touched(self):
        d = self.data
        for i in range(d.ncon):
            c = d.contact[i]
            g = {c.geom1, c.geom2}
            if self.cube_gid in g and g & self.pad_gids:
                return True
        return False

    def success_now(self):
        speed = np.linalg.norm(self.data.qvel[self.cube_vadr:self.cube_vadr + 3])
        dist = np.linalg.norm(self.cube_pos() - self.target)
        return dist < SUCCESS_DIST and speed < 0.05 and not self.cube_touched()

    # perturbation hooks used by chaos.py
    def kick_cube(self, vel_xyz):
        self.data.qvel[self.cube_vadr:self.cube_vadr + 3] += vel_xyz

    def set_grasp_friction(self, mu):
        for g in [self.cube_gid, *self.pad_gids]:
            self.model.geom_friction[g, 0] = mu


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=3)
    args = p.parse_args()
    env = MurphyEnv(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    for ep in range(args.episodes):
        o = env.reset(seed=args.seed + ep)
        print(f"ep {ep} reset obs: {np.round(o, 3)}")
        done, R = False, 0
        while not done:
            o, r, done, info = env.step(rng.uniform(-1, 1, 4))
        print(f"  random policy: steps={info['t']} success={info['success']} cube={np.round(o[4:7], 3)}")
    print("obs_dim", OBS_DIM, "act_dim", ACT_DIM)
