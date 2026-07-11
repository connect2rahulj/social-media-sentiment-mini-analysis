# Social Media Sentiment Mini-Analysis

A beginner-to-intermediate Python project that analyzes the sentiment of sample social media posts. It uses **TextBlob** for sentiment scoring, **Pandas** for data manipulation, and **Matplotlib/Seaborn** for visualizations.

## What It Does

- Loads a CSV of sample social media posts
- Scores each post for **polarity** (positive/negative) and **subjectivity**
- Classifies posts as Positive, Negative, or Neutral
- Generates visualizations:
  - Sentiment distribution bar chart
  - Polarity score histogram
  - Polarity over time line chart

## Project Structure

```
social-media-sentiment/
├── README.md
├── requirements.txt
├── data/
│   └── posts.csv
└── analysis.py
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/social-media-sentiment.git
cd social-media-sentiment
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

You may also need to download the TextBlob corpora:

```bash
python -m textblob.download_corpora
```

### 3. Run the analysis

```bash
python analysis.py
```

Charts will be saved as PNG files in the project directory.

## Dependencies

- Python 3.8+
- pandas
- textblob
- matplotlib
- seaborn

## Sample Output

The script prints a summary table to the terminal and saves three chart images:
- `sentiment_distribution.png`
- `polarity_histogram.png`
- `polarity_over_time.png`

## Skills Demonstrated

- Python scripting
- Data loading and manipulation with Pandas
- NLP basics with TextBlob
- Data visualization with Matplotlib and Seaborn