import pandas as pd
import matplotlib.pyplot as plt

# 1. Read data from CSV
csv_file = "TwoMoons.csv"  # Replace with your file path
df = pd.read_csv(csv_file)

# 2. Create the plot
plt.figure(figsize=(10, 6))

# Define a color palette similar to Seaborn's default
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# Plot each algorithm's loss
for i, column in enumerate(df.columns):
    if column != "Step" and column.endswith("_loss"):
        plt.plot(
            df["Step"] + 1,
            df[column],
            label=column.replace("_loss", ""),
            marker="o",
            color=colors[i % len(colors)],
            linewidth=2,
        )

# 3. Style the plot to look like Seaborn
plt.style.use("seaborn")  # Use a clean style similar to Seaborn

plt.title("Test Loss (after Quantization)", fontsize=14, pad=20)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Test Loss (Quantized)", fontsize=12)
plt.legend(title="Models", title_fontsize=12, bbox_to_anchor=(0.99, 0.99), frameon=True)
plt.grid(True, alpha=0.3)
plt.ylim(0, 0.65)  # Set y-axis limits
plt.tight_layout()
plt.savefig("loss_comparison.png", dpi=300)  # Optional: save the plot
plt.show()
