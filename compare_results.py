import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# --- CONFIG ---
MAP_ELITES_FILE = "map_elites_ant_results.pkl"
MUSE_FILE = "muse_ant_results.pkl"
GRID_RANGE = 30  # +/- 30 meters

def load_data(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"⚠️ Could not find {filename}. Skipping.")
        return None

def plot_comparison():
    me_data = load_data(MAP_ELITES_FILE)
    muse_data = load_data(MUSE_FILE)

    if me_data is None or muse_data is None:
        return

    # Extract Descriptors
    # MAP-Elites descriptors are typically just the list of (x,y)
    me_desc = me_data["final_descriptors"] 
    
    # MUSE descriptors might be stored differently depending on exact script version
    # The script saved 'final_descriptors' directly
    muse_desc = muse_data["final_descriptors"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # --- PLOT 1: MAP-ELITES ---
    ax = axes[0]
    ax.set_title(f"MAP-Elites (Baseline)\nArchive Size: {len(me_desc)}", fontsize=14)
    # Draw limits
    ax.add_patch(plt.Rectangle((-GRID_RANGE, -GRID_RANGE), 2*GRID_RANGE, 2*GRID_RANGE, 
                               fill=False, color='red', linestyle='--', label="Grid Boundary"))
    
    # Scatter points (Color by radius/distance)
    dists = np.linalg.norm(me_desc, axis=1)
    sc = ax.scatter(me_desc[:, 0], me_desc[:, 1], c=dists, cmap='viridis', s=10, alpha=0.7)
    
    ax.set_xlim(-35, 35); ax.set_ylim(-35, 35)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # --- PLOT 2: MUSE-QD ---
    ax = axes[1]
    ax.set_title(f"MUSE-QD (Your Engine)\nArchive Size: {len(muse_desc)}", fontsize=14)
    ax.add_patch(plt.Rectangle((-GRID_RANGE, -GRID_RANGE), 2*GRID_RANGE, 2*GRID_RANGE, 
                               fill=False, color='red', linestyle='--', label="Grid Boundary"))
    
    dists = np.linalg.norm(muse_desc, axis=1)
    sc = ax.scatter(muse_desc[:, 0], muse_desc[:, 1], c=dists, cmap='viridis', s=10, alpha=0.7)
    
    ax.set_xlim(-35, 35); ax.set_ylim(-35, 35)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Colorbar
    cbar = fig.colorbar(sc, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
    cbar.set_label("Distance from Start (Fitness Proxy)", fontsize=12)

    plt.suptitle("Comparison of Reached States (Ant-v4)", fontsize=18)
    plt.show()

    # --- METRICS ---
    print("\n📊 FINAL METRICS")
    print(f"{'Metric':<20} | {'MAP-Elites':<12} | {'MUSE-QD':<12}")
    print("-" * 50)
    print(f"{'Archive Size':<20} | {len(me_desc):<12} | {len(muse_desc):<12}")
    
    # Max Distance Reached
    me_max_dist = np.max(np.linalg.norm(me_desc, axis=1))
    muse_max_dist = np.max(np.linalg.norm(muse_desc, axis=1))
    print(f"{'Max Dist (m)':<20} | {me_max_dist:.2f}{'':<8} | {muse_max_dist:.2f}")

    # Coverage Area (Approximation using Convex Hull could be better, but simple variance works)
    me_cov = np.var(me_desc, axis=0).sum()
    muse_cov = np.var(muse_desc, axis=0).sum()
    print(f"{'Variance (Spread)':<20} | {me_cov:.2f}{'':<8} | {muse_cov:.2f}")

if __name__ == "__main__":
    plot_comparison()
