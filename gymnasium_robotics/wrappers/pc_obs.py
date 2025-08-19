import gymnasium as gym
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
from mujoco import mj_name2id, mjtObj  # from DeepMind mujoco


def depth_gl_to_meters(depth_gl, znear, zfar):
    # Convert OpenGL depth [0,1] to metric depth using near/far planes.
    depth_gl = np.clip(depth_gl, 1e-6, 1.0 - 1e-6)
    return (2.0 * znear * zfar) / (
        zfar + znear - (2.0 * depth_gl - 1.0) * (zfar - znear)
    )


def intrinsics_from_fovy(fovy_deg, W, H):
    fovy = np.deg2rad(float(fovy_deg))
    fy = 0.5 * H / np.tan(0.5 * fovy)
    fx = fy * (W / H)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    return fx, fy, cx, cy


class PointCloudObsWrapper(gym.ObservationWrapper):
    """
    Requires that observations already contain:
      - 'depth': (H, W) float array (meters or OpenGL depth in [0,1])
      - 'rgb':   (H, W, 3) uint8
    Adds:
      - 'pointcloud': (N, 3) float32 (camera or world frame)
      - 'pc_rgb':     (N, 3) uint8    colors aligned with points
    """

    def __init__(
        self,
        env,
        camera_name=None,
        stride=1,
        world_frame=True,
        near_clip=0.03,
        far_clip=5.0,
    ):
        super().__init__(env)
        self.camera_name = camera_name
        self.stride = int(stride)
        self.world_frame = bool(world_frame)
        self.near_clip = float(near_clip)
        self.far_clip = float(far_clip)

        assert isinstance(self.observation_space, gym.spaces.Dict), (
            "PointCloudObsWrapper expects a Dict obs (wrap after DepthObsWrapper)."
        )
        # Variable-length outputs → declare 0-length shape in spaces
        self.observation_space.spaces["pointcloud"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(0, 3), dtype=np.float32
        )
        self.observation_space.spaces["pc_rgb"] = gym.spaces.Box(
            low=0, high=255, shape=(0, 3), dtype=np.uint8
        )

    # ---------- MuJoCo helpers (model/data) ----------
    def _cam_id(self, model):
        if self.camera_name is None:
            return 0
        cam_id = mj_name2id(model, mjtObj.mjOBJ_CAMERA, self.camera_name)
        if cam_id < 0:
            raise ValueError(f"Camera '{self.camera_name}' not found in model.")
        return cam_id

    def _depth_to_meters(self, model, depth):
        # If depth looks like [0,1], convert from GL depth; otherwise assume meters.
        if np.isfinite(depth).any() and np.nanmax(depth) <= 1.0 + 1e-6:
            znear = float(model.vis.map.znear)
            zfar = float(model.vis.map.zfar)
            return depth_gl_to_meters(depth, znear, zfar)
        return depth

    def _camera_pose_world(self, data, cam_id):
        # World-from-camera rotation & translation
        R_w_c = data.cam_xmat[cam_id].reshape(3, 3).copy()
        t_w_c = data.cam_xpos[cam_id].copy()
        return R_w_c, t_w_c

    # ---------- Backprojection ----------
    def _backproject(self, depth_m, rgb, model, data, cam_id):
        H, W = depth_m.shape
        fx, fy, cx, cy = intrinsics_from_fovy(model.cam_fovy[cam_id], W, H)

        ys = np.arange(0, H, self.stride)
        xs = np.arange(0, W, self.stride)
        xv, yv = np.meshgrid(xs, ys)

        z = depth_m[yv, xv]  # (h, w)

        valid = np.isfinite(z) & (z > 0)
        valid &= (z > self.near_clip) & (z < self.far_clip)
        if not np.any(valid):
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)

        u = xv[valid].astype(np.float32)
        v = yv[valid].astype(np.float32)
        z = z[valid].astype(np.float32)

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        Xc = np.stack([x, y, z], axis=1)  # (N, 3) camera frame
        C = rgb[yv[valid], xv[valid]].astype(np.uint8)

        if self.world_frame:
            R_w_c, t_w_c = self._camera_pose_world(data, cam_id)
            Xw = (R_w_c @ Xc.T).T + t_w_c[None, :]
            return Xw.astype(np.float32, copy=False), C
        else:
            return Xc.astype(np.float32, copy=False), C

    # ---------- Gym ObservationWrapper hook ----------
    def observation(self, obs):
        # Get model/data from the unwrapped env (modern Gymnasium robotics)
        u = self.env.unwrapped
        model = getattr(u, "model", None)
        data = getattr(u, "data", None)
        if model is None or data is None:
            raise RuntimeError(
                "Expected env.unwrapped to expose 'model' and 'data' (mujoco)."
            )

        depth = obs["depth"]
        rgb = obs["rgb"]

        cam_id = self._cam_id(model)
        depth_m = self._depth_to_meters(model, depth)
        pc, col = self._backproject(depth_m, rgb, model, data, cam_id)

        obs["pointcloud"] = pc
        obs["pc_rgb"] = col
        return obs
