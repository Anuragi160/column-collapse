import pickle
from absl import app
from absl import flags
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import animation
import numpy as np

flags.DEFINE_string("rollout_dir", None, help="Directory where rollout.pkl are located")
flags.DEFINE_string("rollout_name", None, help="Name of rollout .pkl file")
flags.DEFINE_integer("step_stride", 3, help="Stride of steps to skip.")
flags.DEFINE_bool("change_yz", False, help="Change y and z axis.")

FLAGS = flags.FLAGS


def main(_):
    with open(f"{FLAGS.rollout_dir}{FLAGS.rollout_name}.pkl", "rb") as f:
        data = pickle.load(f)

    gt = np.concatenate([data["initial_positions"], data["ground_truth_rollout"]], axis=0)
    pred = np.concatenate([data["initial_positions"], data["predicted_rollout"]], axis=0)

    num_steps = gt.shape[0]
    loss = data["loss"].item()

    fig = plt.figure(figsize=(24, 10), facecolor='#0d0d0d')
    fig.subplots_adjust(top=0.90, bottom=0.05, left=0.02, right=0.98, wspace=0.05)
    gs = gridspec.GridSpec(1, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0], projection='3d')
    ax2 = fig.add_subplot(gs[1], projection='3d')
    ax1.set_facecolor('#0d0d0d')
    ax2.set_facecolor('#0d0d0d')

    def style_ax(ax, title, title_color):
        ax.set_xlim([0.0, 0.25])
        ax.set_ylim([0.0, 0.05])
        ax.set_zlim([0.0, 0.32])
        ax.set_xlabel('X', color='#aaaaaa', fontsize=10, labelpad=4)
        ax.set_ylabel('Y', color='#aaaaaa', fontsize=10, labelpad=4)
        ax.set_zlabel('Z', color='#aaaaaa', fontsize=10, labelpad=4)
        ax.tick_params(colors='#aaaaaa', labelsize=8)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#333333')
        ax.yaxis.pane.set_edgecolor('#333333')
        ax.zaxis.pane.set_edgecolor('#333333')
        ax.grid(True, alpha=0.15, color='#444444')
        ax.set_title(title, color=title_color, fontsize=17,
                     fontweight='bold', pad=14, fontfamily='monospace')
        ax.set_box_aspect([5, 1, 6.4])
        ax.view_init(elev=25, azim=45)

    def get_colors_dem(z_vals):
        norm = np.clip(z_vals / 0.32, 0, 1)
        r = np.ones_like(norm)
        g = 1.0 - norm * 0.85
        b = np.zeros_like(norm)
        return np.stack([r, g, b], axis=1)

    def get_colors_gns(z_vals):
        norm = np.clip(z_vals / 0.32, 0, 1)
        r = norm
        g = 1.0 - norm * 0.7
        b = np.ones_like(norm)
        return np.stack([r, g, b], axis=1)

    def animate(i):
        ax1.cla()
        ax2.cla()

        style_ax(ax1, '⬤  Reality  (DEM)', '#00d4ff')
        style_ax(ax2, '⬤  GNS Prediction', '#ff6b6b')

        c_gt = get_colors_dem(gt[i, :, 2])
        ax1.scatter(gt[i, :, 0], gt[i, :, 1], gt[i, :, 2],
                    c=c_gt, s=35, alpha=0.95, edgecolors='none', depthshade=True)

        c_pred = get_colors_gns(pred[i, :, 2])
        ax2.scatter(pred[i, :, 0], pred[i, :, 1], pred[i, :, 2],
                    c=c_pred, s=35, alpha=0.95, edgecolors='none', depthshade=True)

        fig.suptitle(
            f'Graph Network Simulator  —  Step {i}/{num_steps}  |  MSE: {loss:.2e}',
            color='white', fontsize=15, fontweight='bold',
            fontfamily='monospace', y=0.97)

        print(f"Render step {i}/{num_steps}")

    ani = animation.FuncAnimation(
        fig, animate,
        frames=np.arange(0, num_steps, FLAGS.step_stride),
        interval=50)

    out = f'{FLAGS.rollout_dir}{FLAGS.rollout_name}_enhanced.gif'
    ani.save(out, dpi=120, fps=20, writer='pillow',
             savefig_kwargs={'facecolor': '#0d0d0d'})
    print(f"Saved to: {out}")


if __name__ == '__main__':
    app.run(main)
