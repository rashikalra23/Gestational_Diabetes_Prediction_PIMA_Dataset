import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def draw_box(ax, x, y, width, height, text, color):
    """Draws a rectangular box with text."""
    box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.3", edgecolor="black", facecolor=color)
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=10, wrap=True)

def draw_arrow(ax, x, y_start, y_end):
    """Draws a smooth, centered arrow."""
    arrow = FancyArrowPatch((x, y_start), (x, y_end), arrowstyle="->", color="black", mutation_scale=15, linewidth=1)
    ax.add_patch(arrow)

# Create the figure
fig, ax = plt.subplots(figsize=(6, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 20)
ax.axis("off")

# Draw the blocks

draw_box(ax, 2, 18, 6, 0.5, "Data Collection", "lightblue")
draw_box(ax, 2, 16, 6, 0.5, "Data Preprocessing \nHandling Missing and Invalid Values, Normalization, Handling Class Imbalance", "lightgreen")
draw_box(ax, 2, 14, 6, 0.5, "Dense Layer 1\nUnits: 32 Activation: ReLU", "wheat")
draw_box(ax, 2, 12, 6, 0.5, "Dense Layer 2\nUnits: 16 Activation: ReLU", "lightgray")
draw_box(ax, 2, 10, 6, 0.5, "Output Layer\nUnits: 1 Activation: sigmoid", "lightblue")
draw_box(ax, 2, 8, 6, 0.5, "Compile Model\nOptimizer: Adam, Loss: Binary Crossentropy, Metric: Accuracy", "lightgreen")
draw_box(ax, 2, 6, 6, 0.5, "Train Model\nEpochs: 50, Batch Size: 16, Validation Split: 20%", "wheat")
draw_box(ax, 2, 4, 6, 0.5, "Model Evaluation\n(Accuracy, Precision, Recall, F1-score, ROC-AUC)", "lightgray")

# Draw the arrows
arrow_x = 5  # Center of the blocks
draw_arrow(ax, arrow_x, 17, 16.5)
draw_arrow(ax, arrow_x, 15, 14.5)
draw_arrow(ax, arrow_x, 13, 12.5)
draw_arrow(ax, arrow_x, 11, 10.5)
draw_arrow(ax, arrow_x, 9, 8.5)
draw_arrow(ax, arrow_x, 7, 6.5)

# Show the flowchart
plt.tight_layout()
plt.show()
