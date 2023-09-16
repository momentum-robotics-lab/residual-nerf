import numpy as np
import matplotlib.pyplot as plt

def compute_color(alpha):
    return (1.0 - alpha, alpha, 0)

# Generate 256 alphas between 0 and 1
alphas = np.linspace(0, 1, 256)

# Compute the corresponding colors for each alpha
colors = [compute_color(alpha) for alpha in alphas]

# Plotting
fig, ax = plt.subplots(figsize=(8, 2), 
                        subplot_kw=dict(xticks=[], yticks=[], frame_on=False))
ax.imshow([colors], extent=[0, 1, 0, 1], aspect='auto')
ax.set_xlabel("Alpha")
plt.savefig("viz_alpha.pdf",  bbox_inches='tight')