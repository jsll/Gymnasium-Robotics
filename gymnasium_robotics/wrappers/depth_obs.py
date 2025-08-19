import gymnasium as gym
import numpy as np


class DepthObsWrapper(gym.ObservationWrapper):
    """
    Adds a single-channel depth image to the observation dict under key 'depth'.
    Tries the modern Gymnasium MuJoCo renderer first, then falls back to mujoco_py.
    """

    def __init__(self, env, width=128, height=128, camera_name=None, camera_id=-1):
        super().__init__(env)
        self.width, self.height = int(width), int(height)
        self.camera_name, self.camera_id = camera_name, int(camera_id)

        # If the base obs is a dict (goal-based), extend it; otherwise make it a dict.
        if isinstance(self.observation_space, gym.spaces.Dict):
            depth_space = gym.spaces.Box(
                low=0.0, high=np.inf, shape=(self.height, self.width), dtype=np.float32
            )
            self.observation_space = gym.spaces.Dict(
                {
                    **self.observation_space.spaces,
                    "depth": depth_space,
                }
            )
        else:
            # Not typical for robotics goal envs, but handle classic control just in case.
            self.observation_space = gym.spaces.Dict(
                {
                    "observation": self.observation_space,
                    "achieved_goal": gym.spaces.Box(
                        low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32
                    ),
                    "desired_goal": gym.spaces.Box(
                        low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32
                    ),
                    "depth": gym.spaces.Box(
                        low=0.0,
                        high=np.inf,
                        shape=(self.height, self.width),
                        dtype=np.float32,
                    ),
                }
            )

    def _render_rgb_depth(self):
        u = self.env.unwrapped

        # Path A: Gymnasium MujocoRenderer
        if hasattr(u, "mujoco_renderer") and u.mujoco_renderer is not None:
            r = u.mujoco_renderer

            # Select camera if requested
            try:
                if self.camera_name is not None or self.camera_id != -1:
                    # Gymnasium exposes a convenience setter
                    r.set_camera(
                        camera_name=self.camera_name
                        if self.camera_name is not None
                        else None,
                        camera_id=None if self.camera_id == -1 else self.camera_id,
                    )
            except TypeError:
                # Older versions may only accept one of the arguments;
                # try the available ones without breaking.
                if self.camera_name is not None:
                    r.set_camera(camera_name=self.camera_name)
                elif self.camera_id != -1:
                    r.set_camera(camera_id=self.camera_id)

            # Now render without kwargs
            rgb = r.render("rgb_array")
            depth = r.render("depth_array")
            return rgb, depth

        # Path B: mujoco_py-style API (legacy)
        if hasattr(u, "sim") and hasattr(u.sim, "render"):
            out = u.sim.render(
                width=self.width,
                height=self.height,
                camera_name=self.camera_name,
                depth=True,
            )
            if isinstance(out, (tuple, list)) and len(out) == 2:
                return out[0], out[1]

        raise RuntimeError("No compatible renderer found for RGB+Depth.")

    def observation(self, obs):
        rgb, depth = self._render_rgb_depth()
        # Ensure expected shapes/dtypes
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        depth = depth.astype(np.float32, copy=False)
        # Optionally, you could normalize here; we leave raw MuJoCo depth.
        if isinstance(obs, dict):
            obs["depth"] = depth
            obs["rgb"] = rgb
            return obs
        else:
            return {
                "observation": obs,
                "achieved_goal": np.array([], dtype=np.float32),
                "desired_goal": np.array([], dtype=np.float32),
                "depth": depth,
                "rgb": rgb,
            }
