import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

import gymnasium_robotics as gr


def plot_pointcloud(pc, colors=None, title="Point Cloud", sample=5000):
    """
    pc: (N,3) numpy array of XYZ points
    colors: (N,3) uint8 or float array of RGB values (optional)
    sample: max number of points to plot for speed
    """
    if pc.size == 0:
        print("Empty point cloud")
        return

    # Subsample if too many points
    if pc.shape[0] > sample:
        idx = np.random.choice(pc.shape[0], sample, replace=False)
        pc = pc[idx]
        if colors is not None:
            colors = colors[idx]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    if colors is not None:
        ax.scatter(
            pc[:, 0],
            pc[:, 1],
            pc[:, 2],
            c=colors / 255.0 if colors.dtype != float else colors,
            s=1,
        )
    else:
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, c="blue")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    plt.show()


env = gym.make("FetchReach-v4", render_mode="rgb_array")
env = gr.DepthObsWrapper(env, width=128, height=128, camera_name=None)
env = gr.PointCloudObsWrapper(
    env,
    camera_name="external_camera_0",
    stride=1,
    world_frame=True,
    near_clip=0.03,
    far_clip=5.0,
)
# u = env.unwrapped
# model = getattr(u, "model", None)

# print(type(model))
# a = gru.mujoco_utils.extract_mj_names(model, mjtObj.mjOBJ_CAMERA)
obs, info = env.reset()
pc = obs["pointcloud"]  # (N,3) float32
pc_rgb = obs["pc_rgb"]  # (N,3) uint8
plot_pointcloud(pc, colors=pc_rgb, title="FetchReach point cloud")
