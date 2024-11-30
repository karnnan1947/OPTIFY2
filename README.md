
# Sentiment Analysis Using Logistic Regression

## Overview
This project implements a machine learning pipeline for **sentiment analysis** of movie reviews using **logistic regression**. The program trains a model on IMDB reviews to classify input reviews as either **Positive** or **Negative**. It features a real-time user interaction loop for predicting sentiments.

---

## Features
- **Text Classification**: Predicts the sentiment of movie reviews as positive or negative.
- **TF-IDF Vectorization**: Converts text data into numerical format for processing.
- **Logistic Regression**: Utilizes a simple and efficient binary classification algorithm.
- **Interactive Predictions**: Accepts user reviews and outputs sentiment predictions in real-time.
- **Performance Evaluation** (Optional): Measures accuracy and generates a classification report.

---

## Dataset
The program uses the **IMDB movie reviews dataset**:
- **Positive Reviews**: Located in `aclImdb/train/pos/`.
- **Negative Reviews**: Located in `aclImdb/train/neg/`.

---

## Requirements
Install the following libraries before running the program:
- `pandas`
- `scikit-learn`

You can install them via pip:
```bash
pip install pandas scikit-learn
```

---

## How It Works
1. **Load Data**:
   - Reads `.txt` files from directories containing positive and negative reviews.
   - Labels positive reviews as `1` and negative reviews as `0`.

2. **Data Preprocessing**:
   - Combines reviews and labels into a single DataFrame.
   - Splits the data into training (80%) and testing (20%) sets.

3. **Vectorization**:
   - Uses `TfidfVectorizer` to convert review texts into numerical representations.
   - Removes stop words and limits the vocabulary to the top 5,000 words.

4. **Model Training**:
   - Trains a logistic regression model on the vectorized training data.

5. **Prediction**:
   - Accepts user input in real-time and predicts whether the review is positive or negative.
   - Outputs the sentiment as "Positive" or "Negative".

6. **Model Evaluation**:
   - Evaluates the model on test data using accuracy and classification metrics.

---

## How to Run
1. Update file paths for the dataset:
   - Replace `t_pos` and `t_neg` with the correct paths to your dataset.
2. Execute the program:
   ```bash
   python sentiment_analysis.py
   ```
3. Enter a movie review when prompted:
   - Type a review (e.g., "The movie was fantastic!") to get its sentiment.
   - Type `exit` to terminate the program.

---

## Example Usage
```bash
sentiment analysis: enter a movie review(or type 'exit'):
Your review: The movie was amazing!
Predicted Sentiment: Positive

Your review: The plot was boring and predictable.
Predicted Sentiment: Negative

Your review: exit
Exiting...
```

---

## Evaluation
Uncomment the following section to evaluate the model:
```python
# Predict on the test set
Y_pred = model.predict(X_test_vec)

# Accuracy score
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Classification report
print("Classification Report:")
print(classification_report(Y_test, Y_pred))
```

---

## Performance Metrics
- **Accuracy**: Percentage of correct predictions.
- **Classification Report**:
  - **Precision**: Proportion of true positives among predicted positives.
  - **Recall**: Proportion of actual positives correctly identified.
  - **F1-Score**: Harmonic mean of precision and recall.

