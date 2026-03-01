import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ─── CONFIG ───────────────────────────────────────────
ROLLOUT_PATH = '/home/vision/Desktop/DEM-ML-250/rollouts/rollout_final_ex0.pkl'
OUTPUT_DIR   = '/home/vision/Desktop/DEM-ML-250/analysis/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── LOAD ROLLOUT ──────────────────────────────────────
with open(ROLLOUT_PATH, 'rb') as f:
    data = pickle.load(f)

gt    = np.concatenate([data['initial_positions'], data['ground_truth_rollout']], axis=0)
pred  = np.concatenate([data['initial_positions'], data['predicted_rollout']], axis=0)
metadata      = data['metadata']
num_steps     = gt.shape[0]
num_particles = gt.shape[1]

print(f"Steps: {num_steps}, Particles: {num_particles}")

# ─── 1. MSE PER TIMESTEP ──────────────────────────────
mse_per_step = np.mean((pred - gt)**2, axis=(1, 2))
print(f"Overall MSE:    {np.mean(mse_per_step):.6f}")
print(f"Final step MSE: {mse_per_step[-1]:.6f}")

# ─── 2. SPEEDUP ───────────────────────────────────────
# GNS: actual GPU rollout — 1994 steps in 18 seconds
gns_time = 18.0 / 1994.0

# DEM actual wall-clock: run_all.py took ~120 minutes for 26 trajectories x 2000 frames
# Change 120 to your actual run_all.py duration in minutes
dem_total_minutes = 120.0
dem_total_trajectories = 26
dem_total_frames = dem_total_trajectories * 2000
dem_time_per_frame = (dem_total_minutes * 60.0) / dem_total_frames

speedup = dem_time_per_frame / gns_time

print(f"\nGNS time per step:  {gns_time*1000:.3f} ms")
print(f"DEM time per frame: {dem_time_per_frame*1000:.1f} ms")
print(f"Speedup (GNS/DEM):  {speedup:.1f}x")

# ─── 3. PLOTS ─────────────────────────────────────────
fig = plt.figure(figsize=(18, 12), facecolor='#0d0d0d')
fig.suptitle('GNS vs DEM — Performance Analysis (250 Particles)',
             color='white', fontsize=15, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Plot 1: Rollout MSE over time
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor('#1a1a2e')
ax1.plot(mse_per_step, color='#00d4ff', linewidth=1.5)
ax1.set_xlabel('Rollout Step', color='white', fontsize=10)
ax1.set_ylabel('MSE', color='white', fontsize=10)
ax1.set_title('Rollout MSE over Time', color='white', fontsize=11, fontweight='bold')
ax1.tick_params(colors='white')
ax1.spines[:].set_color('#444')
ax1.grid(True, alpha=0.2)
ax1.set_yscale('log')

# Plot 2: Speedup bar
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor('#1a1a2e')
bars = ax2.bar(['DEM\n(YADE)', 'GNS\n(Ours)'],
               [dem_time_per_frame * 1000, gns_time * 1000],
               color=['#ff6b6b', '#00d4ff'], width=0.5, edgecolor='none')
ax2.set_ylabel('Time per step (ms)', color='white', fontsize=10)
ax2.set_title(f'Inference Speed\n({speedup:.1f}x Speedup)', color='white',
              fontsize=11, fontweight='bold')
ax2.tick_params(colors='white')
ax2.spines[:].set_color('#444')
ax2.grid(True, alpha=0.2, axis='y')
for bar, val in zip(bars, [dem_time_per_frame*1000, gns_time*1000]):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.01,
             f'{val:.2f}ms', ha='center', va='bottom',
             color='white', fontsize=10, fontweight='bold')

# Plot 3: MSE per dimension
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor('#1a1a2e')
for dim, col, label in zip([0,1,2],
                            ['#ff6b6b','#00ff88','#00d4ff'],
                            ['X','Y','Z']):
    err = np.mean((pred[:,:,dim] - gt[:,:,dim])**2, axis=1)
    ax3.plot(err, color=col, linewidth=1.2, label=label)
ax3.set_xlabel('Rollout Step', color='white', fontsize=10)
ax3.set_ylabel('MSE', color='white', fontsize=10)
ax3.set_title('MSE per Dimension (X/Y/Z)', color='white', fontsize=11, fontweight='bold')
ax3.tick_params(colors='white')
ax3.spines[:].set_color('#444')
ax3.grid(True, alpha=0.2)
ax3.set_yscale('log')
ax3.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)

# Plots 4-6: Particle snapshots
for col_idx, step in enumerate([0, num_steps//2, num_steps-1]):
    ax = fig.add_subplot(gs[1, col_idx], projection='3d')
    ax.set_facecolor('#1a1a2e')
    ax.scatter(gt[step,:,0], gt[step,:,1], gt[step,:,2],
               c='#ff6b6b', s=12, alpha=0.75, label='DEM')
    ax.scatter(pred[step,:,0], pred[step,:,1], pred[step,:,2],
               c='#00d4ff', s=12, alpha=0.75, label='GNS')
    ax.set_xlim([0.0, 0.60])
    ax.set_ylim([0.0, 0.05])
    ax.set_zlim([0.0, 0.32])
    ax.set_title(f'Step {step}  |  MSE={mse_per_step[step]:.5f}',
                 color='white', fontsize=9, fontweight='bold')
    ax.tick_params(colors='white', labelsize=6)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#333')
    ax.yaxis.pane.set_edgecolor('#333')
    ax.zaxis.pane.set_edgecolor('#333')
    ax.grid(True, alpha=0.1)
    ax.view_init(elev=25, azim=45)
    if col_idx == 0:
        ax.legend(facecolor='#1a1a2e', labelcolor='white',
                  fontsize=8, loc='upper right')

out = OUTPUT_DIR + 'gns_vs_dem_analysis.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
print(f"\nSaved to: {out}")

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Particles:         {num_particles}")
print(f"Rollout steps:     {num_steps}")
print(f"Mean MSE:          {np.mean(mse_per_step):.6f}")
print(f"GNS step time:     {gns_time*1000:.3f} ms")
print(f"DEM step time:     {dem_time_per_frame*1000:.1f} ms")
print(f"Speedup:           {speedup:.1f}x")
print(f"Model checkpoint:  540000 steps")
