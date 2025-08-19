import numpy as np
import gymnasium as gym
import gymnasium_robotics as gr
import matplotlib.pyplot as plt
# --- plotting helper ---
def normalize_depth_for_display(depth: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Normalize depth to [0,1] for visualization; ignore inf/NaN."""
    d = depth.astype(np.float32, copy=True)
    mask = np.isfinite(d)
    if not np.any(mask):
        return np.zeros_like(d), 0.0, 0.0
    dmin = float(d[mask].min())
    dmax = float(d[mask].max())
    denom = dmax - dmin if dmax > dmin else 1.0
    d_vis = np.zeros_like(d)
    d_vis[mask] = (d[mask] - dmin) / denom
    return d_vis, dmin, dmax


def show_rgb_depth(rgb: np.ndarray, depth: np.ndarray, title_prefix="Reset"):
    d_vis, dmin, dmax = normalize_depth_for_display(depth)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(rgb)
    axes[0].set_title(f"{title_prefix} — RGB")
    axes[0].axis("off")
    im = axes[1].imshow(d_vis, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"{title_prefix} — Depth (min={dmin:.3f}, max={dmax:.3f})")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="normalized depth")
    plt.tight_layout()
    plt.show()

env = gym.make("FetchReach-v4", render_mode="rgb_array")
env = gr.DepthObsWrapper(env, width=128, height=128, camera_name=None)
obs, info = env.reset()
show_rgb_depth(obs["rgb"], obs["depth"], title_prefix="Reset")
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
show_rgb_depth(obs["rgb"], obs["depth"], title_prefix="Step")

