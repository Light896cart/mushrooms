```markdown
# 🍄 Mushroom Classification Playground — Kaggle 2024

## Overview

Welcome to the **2024 Kaggle Playground Series**! This competition continues the tradition of offering fun and approachable datasets for practicing machine learning skills. Each month features a new challenge, and this one focuses on classifying mushrooms as **edible (e)** or **poisonous (p)**.

---

## 🎯 Your Goal

Predict whether a mushroom is edible or poisonous based on its physical characteristics using synthetic data generated from the original UCI Mushroom dataset.

---

## 📅 Competition Timeline

| Event                      | Date                     |
|---------------------------|--------------------------|
| Start Date                | August 1, 2024           |
| Final Submission Deadline | August 31, 2024          |

All deadlines are at **11:59 PM UTC** unless otherwise noted.

> ⚠️ Note: Organizers reserve the right to update the timeline if necessary.

---

## 🧪 Evaluation Metric

Submissions are evaluated using the **Matthews Correlation Coefficient (MCC)** — a robust metric for binary classification tasks.

---

## 📁 Dataset Description

The training and test datasets were synthetically generated from a deep learning model trained on the [UCI Mushroom dataset](https://archive.ics.uci.edu/ml/datasets/Mushroom ). Feature distributions are similar but not identical to the original dataset.

You're encouraged to use the original dataset for exploration or even to augment your training set.

### Files Included

| File Name             | Description                                                |
|-----------------------|------------------------------------------------------------|
| `train.csv`           | Training data with target column (`class`: e/p)           |
| `test.csv`            | Test data without labels; predict the `class`              |
| `sample_submission.csv` | Sample submission file in required format                 |

---

## 📤 Submission Format

Your submission must be a CSV file with headers and the following format:

```
id,class
3116945,e
3116946,p
3116947,e
...
```

Make sure you predict the class (`e` or `p`) for each row in the test set.

---

## 🔁 About the Tabular Playground Series

The goal of the **Tabular Playground Series** is to provide the Kaggle community with lightweight challenges that allow for quick iteration through different modeling techniques. These competitions typically last a few weeks and offer manageable datasets ideal for:

- Model experimentation
- Feature engineering
- Visualization
- Learning new tools and workflows

---

## 💡 Synthetic Data Notes

This dataset was generated synthetically from real-world data. While it closely resembles the original, there may be categorical values not found in the UCI dataset. Handling these artifacts is part of the challenge.

Please feel free to give feedback on how we can improve future synthetic datasets!
```
