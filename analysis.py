"""
Social Media Sentiment Mini-Analysis
Analyzes sentiment of sample social media posts using TextBlob.
"""

import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# ── 1. Load Data ──────────────────────────────────────────────────────────────

df = pd.read_csv("data/posts.csv", parse_dates=["date"])
print(f"Loaded {len(df)} posts.\n")

# ── 2. Compute Sentiment Scores ───────────────────────────────────────────────

def get_polarity(text):
    return TextBlob(str(text)).sentiment.polarity

def get_subjectivity(text):
    return TextBlob(str(text)).sentiment.subjectivity

def classify_sentiment(polarity):
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

df["polarity"] = df["text"].apply(get_polarity)
df["subjectivity"] = df["text"].apply(get_subjectivity)
df["sentiment"] = df["polarity"].apply(classify_sentiment)

# ── 3. Print Summary Table ────────────────────────────────────────────────────

print("=== Sentiment Summary ===")
print(df[["date", "platform", "sentiment", "polarity", "subjectivity"]].to_string(index=False))

print("\n=== Sentiment Counts ===")
print(df["sentiment"].value_counts().to_string())

print("\n=== Average Polarity by Platform ===")
print(df.groupby("platform")["polarity"].mean().round(3).to_string())

# ── 4. Visualizations ─────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid")

# --- Chart 1: Sentiment Distribution Bar Chart ---
fig, ax = plt.subplots(figsize=(7, 5))
order = ["Positive", "Neutral", "Negative"]
palette = {"Positive": "#4CAF50", "Neutral": "#FFC107", "Negative": "#F44336"}
counts = df["sentiment"].value_counts().reindex(order, fill_value=0)
bars = ax.bar(counts.index, counts.values,
              color=[palette[s] for s in counts.index], edgecolor="white", linewidth=0.8)

for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
            str(val), ha="center", va="bottom", fontweight="bold", fontsize=12)

ax.set_title("Sentiment Distribution of Social Media Posts", fontsize=14, fontweight="bold")
ax.set_xlabel("Sentiment", fontsize=12)
ax.set_ylabel("Number of Posts", fontsize=12)
ax.set_ylim(0, counts.max() + 2)
plt.tight_layout()
plt.savefig("sentiment_distribution.png", dpi=150)
plt.close()
print("\nSaved: sentiment_distribution.png")

# --- Chart 2: Polarity Score Histogram ---
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(df["polarity"], bins=10, color="#2196F3", edgecolor="white", linewidth=0.8)
ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="Neutral boundary")
ax.axvline(df["polarity"].mean(), color="orange", linestyle="-",
           linewidth=1.5, label=f'Mean = {df["polarity"].mean():.2f}')
ax.set_title("Distribution of Polarity Scores", fontsize=14, fontweight="bold")
ax.set_xlabel("Polarity Score", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig("polarity_histogram.png", dpi=150)
plt.close()
print("Saved: polarity_histogram.png")

# --- Chart 3: Polarity Over Time ---
df_sorted = df.sort_values("date")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df_sorted["date"], df_sorted["polarity"],
        marker="o", color="#673AB7", linewidth=1.5, markersize=6, label="Polarity")
ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Neutral (0)")

# Color the marker faces by sentiment
color_map = {"Positive": "#4CAF50", "Neutral": "#FFC107", "Negative": "#F44336"}
for _, row in df_sorted.iterrows():
    ax.plot(row["date"], row["polarity"], "o",
            color=color_map[row["sentiment"]], markersize=8, zorder=5)

ax.set_title("Polarity Score Over Time", fontsize=14, fontweight="bold")
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Polarity Score", fontsize=12)
ax.tick_params(axis="x", rotation=45)

# Manual legend for sentiment colors
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=s) for s, c in color_map.items()]
ax.legend(handles=legend_elements, title="Sentiment", loc="upper right")

plt.tight_layout()
plt.savefig("polarity_over_time.png", dpi=150)
plt.close()
print("Saved: polarity_over_time.png")

print("\nAnalysis complete!")