import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Load dataset
file_path = "modified_PCOS.csv"  # Change to your actual file path
df = pd.read_csv(file_path)

# Drop unnecessary columns
df.drop(columns=['Sl. No', 'Patient File No'], errors='ignore', inplace=True)

# Handle missing values
imputer = SimpleImputer(strategy="mean")
df.iloc[:, 3:] = imputer.fit_transform(df.iloc[:, 3:])  

# Encode categorical variables
label_encoders = {}
for col in ['PCOS (Y/N)', 'Cycle(R/I)', 'Pregnant(Y/N)', 'Weight gain(Y/N)',
            'hair growth(Y/N)', 'Skin darkening (Y/N)', 'Hair loss(Y/N)', 'Pimples(Y/N)', 
            'Fast food (Y/N)', 'Reg.Exercise(Y/N)']:
    
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features (X) and target (y)
X = df.drop(columns=['PCOS (Y/N)'])
y = df['PCOS (Y/N)']

# Standardize numerical features
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X[X.columns])

# Implement K-Fold Cross-Validation
kfold = KFold(n_splits=10, shuffle=True, random_state=42)  # 10-Fold Cross-Validation

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform Cross-Validation
cv_scores = cross_val_score(rf_model, X, y, cv=kfold, scoring='accuracy')

# Print Results
print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean Accuracy: {np.mean(cv_scores):.2f}")
print(f"Standard Deviation: {np.std(cv_scores):.2f}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Load dataset
file_path = "modified_PCOS.csv"  # Change to your actual file path
df = pd.read_csv(file_path)

# Drop unnecessary columns
df.drop(columns=['Sl. No', 'Patient File No'], errors='ignore', inplace=True)

# Handle missing values
imputer = SimpleImputer(strategy="mean")
df.iloc[:, 3:] = imputer.fit_transform(df.iloc[:, 3:])  

# Encode categorical variables
label_encoders = {}
for col in ['PCOS (Y/N)', 'Cycle(R/I)', 'Pregnant(Y/N)', 'Weight gain(Y/N)',
            'hair growth(Y/N)', 'Skin darkening (Y/N)', 'Hair loss(Y/N)', 'Pimples(Y/N)', 
            'Fast food (Y/N)', 'Reg.Exercise(Y/N)']:
    
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features (X) and target (y)
X = df.drop(columns=['PCOS (Y/N)'])
y = df['PCOS (Y/N)']

# Standardize numerical features
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X[X.columns])

# Split dataset into train & test sets (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
svm_model = SVC(kernel='linear', C=1.0)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train models
svm_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_svm = svm_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# Calculate performance metrics
def evaluate_model(y_test, y_pred, model_name):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return [model_name, accuracy, precision, recall, f1]

# Store results in a DataFrame
results = pd.DataFrame([
    evaluate_model(y_test, y_pred_svm, "SVM"),
    evaluate_model(y_test, y_pred_rf, "Random Forest")
], columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])

# Print results
print("\nPerformance Comparison:")
print(results)

# Plot Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, model_name, y_pred in zip(axes, ["SVM", "Random Forest"], [y_pred_svm, y_pred_rf]):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No PCOS", "PCOS"], yticklabels=["No PCOS", "PCOS"], ax=ax)
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

plt.tight_layout()
plt.show()
