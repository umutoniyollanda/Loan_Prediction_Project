# Loan Approval Prediction 

## Introduction

With the advancement in technology, there are so many enhancements in the banking sector also. The number of applications is increasing every day for loan approval. There are some bank policies that they have to consider while selecting an applicant for loan approval. Based
on some parameters, the bank has to decide which one is best for approval. It is tough and risky to check out manually every
person and then recommended for loan approval. In this work, we use a machine learning technique that will predict
the person who is reliable for a loan, based on the previous record of the person whom the loan amount is accredited
before. This work’s primary objective is to predict whether the loan approval to a specific individual is safe or not.
This notebook walks through the preprocessing, exploratory analysis, model training, and evaluation stages using the [Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset) from Kaggle.

## Data Understanding and Exploring

The dataset consists of two files: `train.csv` (with target variable `Loan_Status`) and `test.csv` (without target labels). Each record corresponds to a loan application and includes the following features:

| Column Name        | Description |
|--------------------|-------------|
| `Loan_ID`          | Unique Loan Identifier |
| `Gender`           | Applicant’s gender |
| `Married`          | Marital status |
| `Dependents`       | Number of dependents |
| `Education`        | Educational background |
| `Self_Employed`    | Self-employment status |
| `ApplicantIncome`  | Monthly income of the applicant |
| `CoapplicantIncome`| Income of co-applicant |
| `LoanAmount`       | Loan amount in thousands |
| `Loan_Amount_Term` | Term of loan (in months) |
| `Credit_History`   | Credit history meets guidelines (1: Yes, 0: No) |
| `Property_Area`    | Urban, Semiurban, or Rural |
| `Loan_Status`      | Target variable (Y = approved, N = not approved) |

## Data Preprocessing

Several steps were undertaken to clean and transform the data before modeling:
### Missing values (before cleaning)
The following table shows the number of missing values in each column before cleaning:
#### In Training dataset:
![image.png](attachment:image.png)
#### In Testing dataset:
![image-2.png](attachment:image-2.png)
### Handling Missing Values


- Imputed missing categorical values using mode.
- Imputed `LoanAmount` using median.
- Imputed `Loan_Amount_Term` and `Credit_History` using most frequent values.
### After Handling Missing Values 
#### Training dataset with zero null values
![image-3.png](attachment:image-3.png)
#### Testing dataset with zero null values
![image-4.png](attachment:image-4.png)

### Feature Engineering

- Encoded categorical variables using Label Encoding.

## Exploratory Data Analysis (EDA)

Several plots and statistical summaries were used to understand relationships between:

- Income and loan approval
- Credit history and approval likelihood
- Education and loan amount
- Property area vs approval rate

Visualizations confirmed that credit history and total income are strong indicators of loan approval.

##  Model Training and Evaluation

To evaluate the performance of different machine learning algorithms, we trained and compared two classification models:

###  Logistic Regression

- Logistic Regression is a popular and very useful algorithm
of machine learning for classification problems. The advantage
of logistic regression is that it is a predictive analysis. It is
used for description of data and use to explain relationship
between a single binary variable and single or multiple
nominal, ordinal and ration level variables which are
independent in nature. 
- In this project, we initialized the model with `max_iter=1000` to ensure convergence.
- The model was trained on the training split and evaluated using precision, recall, F1-score, and accuracy metrics.

###  Random Forest Classifier

- Random Forest is an ensemble method that builds multiple decision trees and merges them for better accuracy and robustness.
- It handles non-linear relationships better and is less prone to overfitting compared to a single decision tree.
- Like Logistic Regression, Random Forest was evaluated using a classification report and confusion matrix.

###  Performance Evaluation

We used the `classification_report` from Scikit-learn to evaluate both models on a validation set. The report includes:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

To print accuracy scores directly, use the following code:
```python
from sklearn.metrics import accuracy_score
print("Logistic Regression Accuracy:", accuracy_score(y_val_split, y_pred))
print("Random Forest Accuracy:", accuracy_score(y_val_split, val_preds))

## Confusion Matrix
Both models were also assessed with confusion matrices to visualize the distribution of true positives, true negatives, false positives, and false negatives. This helps in identifying where the model is making incorrect predictions.
