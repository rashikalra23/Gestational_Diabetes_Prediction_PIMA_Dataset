from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch


def draw_box(ax, text, x, y, width=1.8, height=0.8):
    box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.2", edgecolor="black", facecolor="lightblue")
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, ha="center", va="center", fontsize=8, wrap=True)

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, 6)
ax.set_ylim(-4, 10)
ax.axis('off')

# Draw flowchart boxes
draw_box(ax, "Data Collection", 2, 9)
draw_box(ax, "Data Preprocessing", 2, 7.5)
draw_box(ax, "Handling Missing and Invalid Values", 2, 6)
draw_box(ax, "Normalization", 2, 4.5)
draw_box(ax, "Handling Class Imbalance", 2, 3)
draw_box(ax, "Feature Engineering", 2, 1.5)
draw_box(ax, "Model Building\n(Logistic Regression, Decision Tree, Random Forest,\nNaive Bayes, SVM, Gradient Boosting, XGBoost, AdaBoost)", 2, 0)
draw_box(ax, "Model Evaluation\n(Accuracy, Precision, Recall, F1-score, ROC-AUC)", 2, -1.5)
draw_box(ax, "Hyperparameter Tuning\n(for Best Model)", 2, -3)

# Add arrows with corrected downward direction
arrowprops = dict(arrowstyle="->", color="black", lw=1.5)
ax.annotate("", xy=(3, 8.2), xytext=(3, 8.8), arrowprops=arrowprops)  # Data Collection to Preprocessing
ax.annotate("", xy=(3, 6.7), xytext=(3, 7.3), arrowprops=arrowprops)  # Preprocessing to Missing Values
ax.annotate("", xy=(3, 5.2), xytext=(3, 5.8), arrowprops=arrowprops)  # Missing Values to Normalization
ax.annotate("", xy=(3, 3.7), xytext=(3, 4.3), arrowprops=arrowprops)  # Normalization to Class Imbalance
ax.annotate("", xy=(3, 2.2), xytext=(3, 2.8), arrowprops=arrowprops)  # Class Imbalance to Feature Engineering
ax.annotate("", xy=(3, 0.7), xytext=(3, 1.3), arrowprops=arrowprops)  # Feature Engineering to Model Building
ax.annotate("", xy=(3, -0.8), xytext=(3, -0.2), arrowprops=arrowprops)  # Model Building to Evaluation
ax.annotate("", xy=(3, -2.7), xytext=(3, -2.3), arrowprops=arrowprops)  # Model Evaluation to Hyperparameter Tuning

# Show the corrected flowchart
plt.tight_layout()
plt.show()
