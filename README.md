# Spam-Email-Prediction-Using-Regression
This project implements a spam mail detection system using machine learning techniques. The goal is to build a model that can classify emails as either "spam" or "ham" (not spam) based on their content.

Key Steps:
-Importing Dependencies: Necessary libraries like numpy, pandas, sklearn are imported for data manipulation, model building, and evaluation.

Data Collection and Preprocessing:
-The mail_data.csv dataset is loaded into a pandas DataFrame.
-Null values in the dataset are replaced with empty strings to ensure data consistency.
-The 'Category' column is label encoded, where 'spam' is represented by 0 and 'ham' by 1. This converts the categorical labels into numerical values suitable for machine learning.
-Splitting Data: The dataset is split into training and testing sets (80% for training, 20% for testing) to evaluate the model's performance on unseen data.

-Feature Extraction:
The text data (email messages) is transformed into numerical feature vectors using TfidfVectorizer. This technique converts text into a matrix of TF-IDF features, representing the importance of words in each email.
The target variables (Y_train and Y_test) are converted to integers.

-Training the Machine Learning Model: A Logistic Regression model is trained on the training data (the TF-IDF features and their corresponding labels).

-Evaluating the Trained Model:
The model's accuracy is evaluated on both the training and testing datasets using the accuracy_score metric. This provides insights into how well the model performs on data it has seen and data it hasn't seen.

-Building a Predictive System: A function is created to take a new email message as input, transform it into a feature vector using the trained TfidfVectorizer, and then use the trained Logistic Regression model to predict whether the email is spam or ham.

Technologies Used:

Python
pandas
numpy
scikit-learn
This project demonstrates a basic workflow for building a text classification model, specifically for spam detection.
