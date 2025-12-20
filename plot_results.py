import pickle
import matplotlib.pyplot as plt
import numpy as np

def plot_dashboard():
    with open("muse_results.pkl", "rb") as f:
        data = pickle.load(f)
        
    history = data["history"]
    tips = data["final_tips"]
    
    fig = plt.figure(figsize=(12, 6))
    
    # --- Plot 1: Performance ---
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(history["gen"], history["max_fit"], label="Max Fitness", color="crimson", linewidth=2)
    ax1.plot(history["gen"], history["avg_fit"], label="Avg Fitness", color="gray", linestyle="--")
    ax1.set_xlabel("Generations")
    ax1.set_ylabel("Fitness (1.0 = Target)")
    ax1.set_title("MUSE-QD Convergence")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # --- Plot 2: Behavior Space (The Trap) ---
    ax2 = fig.add_subplot(1, 2, 2)
    
    # Draw Trap (Wall)
    # Wall x=[0.5, 0.55], y=[-0.5, 0.5]
    rect = plt.Rectangle((0.5, -0.5), 0.05, 1.0, color='black', alpha=0.5, label="Trap")
    ax2.add_patch(rect)
    
    # Draw Target
    target = plt.Circle((0.8, 0.0), 0.02, color='green', label="Target")
    ax2.add_patch(target)
    
    # Draw Elites
    # These are the end-effector positions of the survivors
    ax2.scatter(tips[:, 0], tips[:, 1], c=tips[:, 0], cmap="viridis", s=10, alpha=0.6, label="Elites")
    
    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(-1.0, 1.0)
    ax2.set_title("Archive Coverage (End Effectors)")
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig("muse_artifact.png", dpi=150)
    print("🎨 Plot generated: muse_artifact.png")

if __name__ == "__main__":
    plot_dashboard()
