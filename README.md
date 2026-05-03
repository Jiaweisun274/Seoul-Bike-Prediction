# Seoul Bike Sharing Demand Prediction

ML course project at Aalto University. Predicts hourly bike rental demand in Seoul from weather and temporal features, comparing polynomial regression and MLP neural networks.

## Problem

Given weather conditions (temperature, humidity, wind speed, rainfall, snowfall, solar radiation) and temporal features (hour of day, season, holiday), predict how many bikes will be rented in a given hour.

## Approach

Two model families are compared:

**Polynomial Regression** — tested across degrees 0 to 4, with standardised continuous features and one-hot encoded seasons. Hour of day is encoded cyclically (sine/cosine) to capture the circular nature of time.

**MLP Neural Network** — tested across 1, 2, 5, and 10 hidden layers with 64 neurons each. Best architecture selected on validation set, then retrained on train+val before final test evaluation.

Both models use the same feature set and the same 70/15/15 train/validation/test split.

## Feature Engineering

- Cyclical encoding of hour (sin/cos) to avoid artificial discontinuity at midnight
- Binary rain-or-snow flag combined from rainfall and snowfall columns
- PCA applied to correlated weather features (temperature, dew point)
- One-hot encoding of seasons (drop-first)
- Holiday encoded as binary 0/1

## Dataset

Seoul Bike Sharing Demand dataset — 8,760 hourly observations covering one full year.  
Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand)

## Requirements

```
pip install -r requirements.txt
```

Run the notebook: `project.ipynb`
