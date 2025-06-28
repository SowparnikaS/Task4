# Spam Detection using Machine Learning – CodTech Internship Task 4
- This project is developed as part of Task4 for the internship with CodTech.It focuses on implementing a machine learning model using Scikit-learn to detect spam.

## Objective

- Build a predictive model using Scikit-learn
- Classify text messages as spam or ham (not spam)
- Evaluate the model’s accuracy using standard metrics

## Features

- Loads and processes text message data
- Converts text into numerical format using **TF-IDF Vectorization**
- Trains a **Naive Bayes** classifier
- Displays:
  - Overall accuracy
  - Precision, recall, and F1-score
  - Confusion matrix (with heatmap)

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## How to Run

1. Clone or download the Jupyter Notebook / Python script.

2. Install the required libraries:
   pip install pandas scikit-learn matplotlib seaborn
Open the notebook or script and run all cells:

If using Jupyter Notebook: Shift + Enter on each cell

If using Python script: python spam_detector.py

The dataset is loaded automatically from:

https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv

## Sample Output
- Accuracy: ~98%

- Classification Report:

Shows precision, recall, F1-score for both spam and ham

- Graph: Confusion matrix heatmap for visual understanding

## Deliverables
- spam_detector.ipynb – Jupyter Notebook with model implementation

- README.md – Documentation of the project

## Author
- Sowparnika S (Intern at CodTech)
