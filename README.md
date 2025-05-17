# Spam Email Filtering using Machine Learning

This project is a machine learning-based solution for detecting spam emails. It uses Python and the scikit-learn library to build and evaluate a logistic regression model that classifies emails as "spam" or "ham" (not spam) based on their content.

## Project Overview

The goal of this project is to predict whether a given email is spam or not using classic NLP and machine learning techniques. The project includes data preprocessing, feature extraction, model training, evaluation, and prediction.

## Dataset

The dataset used consists of 5,572 labeled emails, each with a category ("spam" or "ham") and a message. The data is loaded from a CSV file named `mail_data.csv`.

**Sample columns:**
- `Category`: spam or ham (later encoded as 0 for spam, 1 for ham)
- `Message`: The content of the email

## Steps and Methodology

### 1. Importing Dependencies

Key libraries:
- `numpy`, `pandas` for data manipulation
- `scikit-learn` for model building and evaluation

### 2. Data Collection & Preprocessing

- Load the dataset into a pandas DataFrame
- Handle missing values by replacing them with empty strings
- Encode labels: spam → 0, ham → 1

### 3. Feature Extraction

- Use `TfidfVectorizer` to convert email text to numerical features, removing English stop words and converting text to lowercase.

### 4. Splitting Data

- The dataset is split into training (80%) and testing (20%) sets.

### 5. Model Building

- A Logistic Regression model is trained on the extracted features.

### 6. Evaluation

- The model achieves high accuracy:
    - Training accuracy: ~96.7%
    - Testing accuracy: ~96.6%

### 7. Prediction System

- You can input a new email message, and the system will predict whether it is spam or ham.

## Example Usage

```python
# Example of predicting a new email
input_mail = ["Congratulations! You've won a free ticket. Reply WIN to claim."]
input_data_features = feature_extraction.transform(input_mail)
prediction = model.predict(input_data_features)

if prediction[0] == 1:
    print('Ham mail')
else:
    print('Spam mail')
```

## Requirements

- Python 3
- pandas
- numpy
- scikit-learn

## Running the Project

1. Clone this repository.
2. Ensure you have the required dependencies installed.
3. Place the `mail_data.csv` file in the appropriate directory.
4. Run the Jupyter notebook:  
   `Project_17_Spam_Mail_Prediction_using_Machine_Learning.ipynb`

## File Structure

- `Project_17_Spam_Mail_Prediction_using_Machine_Learning.ipynb`: Main notebook containing all code and explanations.
- `mail_data.csv`: The dataset (not included in the repo, add separately if needed).

## Results

- Achieves over 96% accuracy on both training and test data.
- Demonstrates a simple, effective approach to spam detection using logistic regression.

---
