---
Title: Mental Health Tracker
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.33.0
python_version: "3.10"
app_file: app.py
pinned: false
---

# Mental Health Sentiment Tracker

An AI-powered web application that analyzes journal entries to detect emotions and mental health risk levels, computing a unified wellness score and visualizing trends over time.

## Features
- **Emotion Detection** — 7-class emotion classifier (joy, sadness, fear, anger, neutral, surprise, disgust) using `j-hartmann/emotion-english-distilroberta-base`
- **Risk Assessment** — Zero-shot mental health risk classification (normal / anxiety / depression / high-risk) with keyword safety override
- **Wellness Score** — Aggregated 0–100 score computed from emotion weights and risk penalties
- **Real-time Alerts** — Color-coded alerts with helpline information for low wellness scores
- **Dashboard** — Plotly charts: wellness trend, emotion timeline, risk distribution
- **SQLite Storage** — All entries persisted locally with timestamps

## Folder Structure
```
mental_health_tracker/
├── app.py                  # Streamlit main application
├── requirements.txt
├── README.md
├── models/
│   ├── emotion_model.py    # Emotion classifier (HuggingFace)
│   └── risk_model.py       # Risk detector (zero-shot BART)
├── utils/
│   ├── aggregator.py       # Wellness score formula
│   ├── database.py         # SQLite read/write
│   └── preprocessor.py     # Text cleaning
└── data/
    └── entries.db          # Auto-created on first run
```

## Tech Stack
| Layer | Technology |
|---|---|
| Frontend | Streamlit + Custom CSS (dark theme) |
| Visualization | Plotly |
| Emotion Model | `j-hartmann/emotion-english-distilroberta-base` |
| Risk Model | `facebook/bart-large-mnli` (zero-shot) |
| Storage | SQLite via `sqlite3` |
| Data | pandas, numpy |

## Setup and Installation

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the application
```bash
streamlit run app.py
```

## Wellness Score Formula
```
emotion_score = (positive_weighted / total_weighted) × 100
risk_penalty  = { normal: 0, anxiety: 20, depression: 45, high_risk: 75 }
wellness      = (emotion_score × 0.6) + ((100 - risk_penalty) × 0.4)
```

## Alert Thresholds
| Score | Level | Action |
|---|---|---|
| 70–100 | 🟢 Good | Encouragement message |
| 45–69 | 🟡 Moderate | Self-care suggestions |
| 0–44 | 🔴 Low | Helpline contact shown |

## Notes
- Models are downloaded automatically from HuggingFace on first run (~1.5 GB)
- The database file is created automatically at `data/entries.db`
- This tool is for personal insight only — **not a substitute for professional care**
