import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIG ---
REMOTE_USER = "laz"
REMOTE_HOST = "phoenix.wordwaves.app"
REMOTE_DIR = "~/mmr-elites"
LOCAL_FILE = "muse_results.pkl"
LAMBDA = 0.8

def get_joint_coords(joints):
    dof = 20
    link_len = 1.0 / dof
    angles = np.cumsum(joints)
    dx = link_len * np.cos(angles)
    dy = link_len * np.sin(angles)
    x = np.concatenate(([0], np.cumsum(dx)))
    y = np.concatenate(([0], np.cumsum(dy)))
    return np.stack([x, y], axis=1)

def analyze():
    with open(LOCAL_FILE, "rb") as f:
        data = pickle.load(f)

    snapshots = data["snapshots"]
    sorted_gens = sorted(snapshots.keys())
    plot_gens = [1, 10, 100, sorted_gens[-1]]
    
    fig, axes = plt.subplots(len(plot_gens), 3, figsize=(15, 16))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    plot_row = 0
    for gen in sorted_gens:
        snap = snapshots[gen]
        pop_fit = snap["fitness"]
        pop_genes = snap["genomes"]
        
        # Pre-calculate all joint descriptors for the population
        # (Batch, 21, 2) -> flattened to (Batch, 42)
        all_coords = np.array([get_joint_coords(g).flatten() for g in pop_genes])
        
        # --- TRUE GREEDY MMR SELECTION ---
        selected_indices = []
        selected_stats = [] # Stores (fit, div, score)
        
        for rank in range(3):
            best_mmr = -np.inf
            best_idx = -1
            best_div = 0
            
            for i in range(len(pop_fit)):
                if i in selected_indices: continue
                
                fit = pop_fit[i]
                if rank == 0:
                    score = fit
                    div = 0
                else:
                    # Marginal distance to the ALREADY SELECTED set
                    current_coords = all_coords[i]
                    selected_coords = all_coords[selected_indices]
                    dists = np.linalg.norm(selected_coords - current_coords, axis=1)
                    div = np.min(dists)
                    score = (LAMBDA * fit) + ((1.0 - LAMBDA) * div)
                
                if score > best_mmr:
                    best_mmr = score
                    best_idx = i
                    best_div = div
            
            selected_indices.append(best_idx)
            selected_stats.append((pop_fit[best_idx], best_div, best_mmr))

        # --- PRINT AUDIT ---
        print(f"\nGeneration {gen} (True MMR Selection):")
        for i, idx in enumerate(selected_indices):
            f, d, s = selected_stats[i]
            d_str = f"{d:.4f}" if i > 0 else "N/A"
            print(f"  Selection #{i+1} | Index: {idx:<4} | Fit: {f:.4f} | Joint-Div: {d_str:<8} | MMR-Score: {s:.4f}")

        # --- PLOTTING ---
        if gen in plot_gens:
            for i, idx in enumerate(selected_indices):
                ax = axes[plot_row][i]
                coords = get_joint_coords(pop_genes[idx])
                jx, jy = coords[:, 0], coords[:, 1]
                
                # Draw Ghosts of ALL previously selected in this gen
                for prev_idx in selected_indices[:i]:
                    p_coords = get_joint_coords(pop_genes[prev_idx])
                    ax.plot(p_coords[:, 0], p_coords[:, 1], 'k--', linewidth=1, alpha=0.15)
                
                # Draw Active
                ax.plot(jx, jy, 'b.-', linewidth=2, markersize=4)
                ax.plot([0.5, 0.5], [-0.25, 0.25], 'r-', linewidth=5, alpha=0.6) # Wall
                ax.scatter([0.8], [0.0], c='lime', s=150, edgecolors='k', zorder=10)
                
                f, d, s = selected_stats[i]
                ax.set_title(f"Gen {gen} | Selection #{i+1}\nMMR Score: {s:.3f}", fontsize=9)
                ax.set_xlim(-0.2, 1.1); ax.set_ylim(-0.6, 0.6)
                ax.set_aspect('equal')
            plot_row += 1

    plt.suptitle(f"MMR Selection Audit (λ={LAMBDA}, Joint-Space Diversity)", fontsize=16)
    plt.show()

if __name__ == "__main__":
    analyze()
