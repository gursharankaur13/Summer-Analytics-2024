import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

# Load the datasets
train_features = pd.read_csv(r'C:\Users\Win 10\OneDrive\Documents\Summer Analytics 2024\training_set_features.csv')
train_labels = pd.read_csv(r'C:\Users\Win 10\OneDrive\Documents\Summer Analytics 2024\training_set_labels.csv')
test_features = pd.read_csv(r'C:\Users\Win 10\OneDrive\Documents\Summer Analytics 2024\test_set_features.csv')
submission_format = pd.read_csv(r'C:\Users\Win 10\OneDrive\Documents\Summer Analytics 2024\submission_format.csv')

# Separate categorical and numerical features
categorical_features = train_features.select_dtypes(include=['object']).columns
numerical_features = train_features.select_dtypes(include=['number']).columns.drop('respondent_id')

# Handle missing values for numerical features
num_imputer = SimpleImputer(strategy='median')
train_features[numerical_features] = num_imputer.fit_transform(train_features[numerical_features])
test_features[numerical_features] = num_imputer.transform(test_features[numerical_features])

# Handle missing values and encode categorical features
cat_imputer = SimpleImputer(strategy='most_frequent')
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
imputed_cat_features = cat_imputer.fit_transform(train_features[categorical_features])
encoded_cat_features = encoder.fit_transform(imputed_cat_features)
imputed_cat_features_test = cat_imputer.transform(test_features[categorical_features])
encoded_cat_features_test = encoder.transform(imputed_cat_features_test)

# Combine numerical and encoded categorical features
processed_features = np.hstack((train_features[numerical_features], encoded_cat_features))
processed_test_features = np.hstack((test_features[numerical_features], encoded_cat_features_test))

# Prepare the labels
labels = train_labels[['xyz_vaccine', 'seasonal_vaccine']]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(processed_features, labels, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
multi_target_rf = MultiOutputClassifier(rf_classifier, n_jobs=-1)
multi_target_rf.fit(X_train, y_train)

# Make predictions on the test set
test_predictions = multi_target_rf.predict_proba(processed_test_features)
test_pred_proba = np.array([pred[:, 1] for pred in test_predictions]).T

# Prepare the submission file
submission_format['xyz_vaccine'] = test_pred_proba[:, 0]
submission_format['seasonal_vaccine'] = test_pred_proba[:, 1]
submission_format.to_csv(r'C:\Users\Win 10\OneDrive\Documents\Summer Analytics 2024\submission.csv', index=False)
print('Submission file created successfully!')
