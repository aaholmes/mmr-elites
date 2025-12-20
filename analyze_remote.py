import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIG ---
REMOTE_USER = "laz"
REMOTE_HOST = "phoenix.wordwaves.app"
REMOTE_DIR = "~/mmr-elites"
LOCAL_FILE = "muse_results.pkl"
LAMBDA = 0.95  # The exact weighting used in experiment

def get_joint_coords(joints):
    """Reconstruct arm stick figure from angles."""
    dof = 20
    link_len = 1.0 / dof
    angles = np.cumsum(joints)
    dx = link_len * np.cos(angles)
    dy = link_len * np.sin(angles)
    x = np.concatenate(([0], np.cumsum(dx)))
    y = np.concatenate(([0], np.cumsum(dy)))
    return x, y

def get_data():
    print(f"📡 Downloading data from {REMOTE_HOST}...")
    cmd = f"scp {REMOTE_USER}@{REMOTE_HOST}:{REMOTE_DIR}/muse_results.pkl {LOCAL_FILE}"
    if os.system(cmd) != 0:
        print("❌ SCP Failed."); exit(1)
    print("✅ Download complete.")

def analyze():
    with open(LOCAL_FILE, "rb") as f:
        data = pickle.load(f)

    snapshots = data["snapshots"]
    sorted_gens = sorted(snapshots.keys())
    
    print(f"\n🔍 MMR SEQUENTIAL ANALYSIS (λ={LAMBDA})")
    print("   We verify the 'Marginal' gain of the Top 3 Elites.")
    print("   Score = λ * Fit + (1-λ) * Dist(Selected_So_Far)")
    print("=" * 110)
    
    # Plot Config: Show Start, Early-Mid, Late-Mid, End
    plot_gens = [1, 10, 100, sorted_gens[-1]]
    fig, axes = plt.subplots(len(plot_gens), 3, figsize=(15, 16))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    plot_row = 0
    
    for gen in sorted_gens:
        snap = snapshots[gen]
        # Get Top 3 purely by Fitness (The "Best" solutions found)
        # We want to see how diverse these "Best" ones actually are.
        top_idx = np.argsort(snap["fitness"])[-3:][::-1]
        
        selected_tips = [] # To store the tips of E1, E2... for distance calc
        
        # Only print text for generations we saved (1-10, then every 20)
        # The snapshot dict already filters this, so we iterate all present keys.
        print(f"\nGeneration {gen}:")
        header = f"  {'Rank':<4} | {'Fit (A)':<8} | {'Div (B)':<8} | {'Total (λA + (1-λ)B)':<20} | {'Tip (x,y)':<15}"
        print(header)
        print("  " + "-" * len(header))
        
        for i, idx in enumerate(top_idx):
            fitness = snap["fitness"][idx]
            tip = snap["tips"][idx]
            
            # --- MMR CALCULATION ---
            if i == 0:
                # Rank 1: First pick. Distance term doesn't exist (or is max).
                # Effectively score is just Fitness.
                div_score = 0.0 # Placeholder
                mmr_score = fitness # Pure exploitation
                dist_str = "N/A (First)"
            else:
                # Rank 2+: Distance to PREVIOUSLY selected in this specific group
                # (Calculating marginal diversity relative to better peers)
                dists = np.linalg.norm(np.array(selected_tips) - tip, axis=1)
                div_score = np.min(dists)
                dist_str = f"{div_score:.4f}"
                
                # The Formula: λ*Fit + (1-λ)*Div
                mmr_score = (LAMBDA * fitness) + ((1.0 - LAMBDA) * div_score)
            
            selected_tips.append(tip)
            
            print(f"  #{i+1:<3} | {fitness:.4f}   | {dist_str:<8} | {mmr_score:.4f}               | ({tip[0]:.2f}, {tip[1]:.2f})")

        # --- PLOTTING ---
        if gen in plot_gens:
            for i, idx in enumerate(top_idx):
                if plot_row >= len(plot_gens): continue
                ax = axes[plot_row][i]
                
                joints = snap["genomes"][idx]
                jx, jy = get_joint_coords(joints)
                
                # Draw Components
                ax.plot(jx, jy, 'b.-', linewidth=2, markersize=4, alpha=0.8, label="Arm") # Arm
                ax.plot([0.5, 0.5], [-0.25, 0.25], 'r-', linewidth=5, alpha=0.6) # Wall
                ax.scatter([0.8], [0.0], c='lime', s=150, edgecolors='k', zorder=10) # Target
                
                # Draw Ghost of Previous Ranks (to visualize diversity)
                if i > 0:
                     # Plot the previous elite as a faint gray line to show difference
                     prev_idx = top_idx[i-1]
                     px, py = get_joint_coords(snap["genomes"][prev_idx])
                     ax.plot(px, py, 'k--', linewidth=1, alpha=0.3, label=f"Rank #{i}")

                # Title & Format
                f_val = snap["fitness"][idx]
                ax.set_title(f"Gen {gen} | Rank #{i+1}\nFit: {f_val:.3f}", fontsize=10)
                ax.set_xlim(-0.2, 1.1)
                ax.set_ylim(-0.6, 0.6)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.2)
                
            plot_row += 1

    print("\n🎨 Visualization Complete. Showing plots...")
    plt.show()

if __name__ == "__main__":
    get_data()
    analyze()
