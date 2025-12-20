import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# --- SSH CONFIG ---
REMOTE_USER = "laz"
REMOTE_HOST = "phoenix.wordwaves.app"
REMOTE_DIR = "~/mmr-elites"
LOCAL_FILE = "muse_results.pkl"

def get_data():
    print(f"📡 Downloading data from {REMOTE_HOST}...")
    # scp laz@phoenix:~/mmr-elites/muse_results.pkl .
    cmd = f"scp {REMOTE_USER}@{REMOTE_HOST}:{REMOTE_DIR}/muse_results.pkl {LOCAL_FILE}"
    ret = os.system(cmd)
    if ret != 0:
        print("❌ SCP Failed. Check your VPN/SSH connection.")
        exit(1)
    print("✅ Download complete.")

def analyze():
    with open(LOCAL_FILE, "rb") as f:
        data = pickle.load(f)

    snapshots = data["snapshots"]
    final_tips = data["final_tips"]
    
    # 1. TEXT TABLE: Coordinate Divergence
    print("\n🔍 DIVERSITY AUDIT: Top 3 Elites Coordinates (x, y)")
    print(f"{'Gen':<6} | {'Elite #1':<22} | {'Elite #2':<22} | {'Elite #3':<22}")
    print("-" * 80)
    
    sorted_gens = sorted(snapshots.keys())
    target = np.array([0.8, 0.0])
    
    # Show every 20 gens to keep table readable
    for gen in sorted_gens[::2]: 
        tips = snapshots[gen]
        # Recalculate fitness (1.0 - dist) to find the elites
        dists = np.linalg.norm(tips - target, axis=1)
        # Sort by distance (ascending) -> Top Elites are first
        top_idx = np.argsort(dists)[:3]
        
        row = f"{gen:<6} | "
        for idx in top_idx:
            x, y = tips[idx]
            # Color code: If y is very different, diversity is working
            row += f"({x:5.2f}, {y:5.2f})       | "
        print(row)

    # 2. PLOT
    plt.figure(figsize=(10, 5))
    
    # Left: The Cloud
    plt.subplot(1, 2, 1)
    plt.scatter([0.8], [0.0], c="lime", s=200, edgecolors="k", zorder=10, label="Target")
    plt.plot([0.5, 0.5], [-0.25, 0.25], 'r-', linewidth=5, label="Wall")
    plt.scatter(final_tips[:, 0], final_tips[:, 1], c="blue", s=10, alpha=0.3)
    plt.title("Final Archive (Blue Cloud)")
    plt.xlim(0, 1.0); plt.ylim(-0.6, 0.6)
    plt.grid(True, alpha=0.3)
    
    # Right: Trajectory of the Winner
    plt.subplot(1, 2, 2)
    best_x, best_y = [], []
    for gen in sorted_gens:
        tips = snapshots[gen]
        dists = np.linalg.norm(tips - target, axis=1)
        best = tips[np.argmin(dists)]
        best_x.append(best[0])
        best_y.append(best[1])
        
    plt.plot(best_x, best_y, 'k.-', alpha=0.5, linewidth=1, label="Best Trace")
    plt.plot([0.5, 0.5], [-0.25, 0.25], 'r-', linewidth=5)
    plt.scatter([0.8], [0.0], c="lime", s=200, edgecolors="k", zorder=10)
    plt.title("Trace of Best Individual")
    plt.xlim(0, 1.0); plt.ylim(-0.6, 0.6)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    get_data()
    analyze()
