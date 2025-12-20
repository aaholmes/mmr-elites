import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load Data
with open("muse_results.pkl", "rb") as f:
    data = pickle.load(f)

history = data["history"]
tips = data["final_tips"]

# --- PLOT 1: Performance ---
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history["gen"], history["max_fit"], label="Max Fitness", color="blue", linewidth=2)
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.title("Convergence Speed")
plt.grid(True, alpha=0.3)
plt.legend()

# --- PLOT 2: The Archive (Behavior Space) ---
plt.subplot(1, 2, 2)
# Draw Target
plt.scatter([0.8], [0.0], c="lime", s=200, label="Target", edgecolors="black", zorder=10)
# Draw Wall (Fixed Geometry)
plt.plot([0.5, 0.5], [-0.25, 0.25], 'r-', linewidth=5, label="Wall")

# Draw Elites (The "Cloud")
plt.scatter(tips[:, 0], tips[:, 1], c="blue", s=10, alpha=0.3, label="Archive (Hand Positions)")

plt.xlim(0.0, 1.0)
plt.ylim(-0.6, 0.6)
plt.title("Diversity: Archive Coverage")
plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig("muse_artifact.png", dpi=150)
print("🎨 Plot saved to 'muse_artifact.png'")
