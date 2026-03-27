import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697) 
  
# data (as pandas dataframes) 
df = predict_students_dropout_and_academic_success.data.original

# The column name for Daytime/evening attendance has a trailing tab/quote in the sample, let's strip whitespace from column names
df.columns = df.columns.str.strip().str.replace('"', '')

# Map Target to Binary classification: 0 for Dropout, 1 for Enrolled/Graduate
target_mapping = {'Dropout': 0, 'Enrolled': 1, 'Graduate': 1}
df['Target'] = df['Target'].map(target_mapping)

# Drop any potential NaNs in Target just in case
df = df.dropna(subset=['Target'])

X = df.drop('Target', axis=1)
y = df['Target']

# 2. Visualize the data (Redundancy and Distribution)
# Correlation matrix to find redundant features
plt.figure(figsize=(12, 10))
corr_matrix = X.corr()
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False)
plt.title('Feature Correlation Matrix (Check for Redundancy)')
plt.tight_layout()
plt.show()

# Normalize features for distribution visualization
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_scaled['Target'] = y.reset_index(drop=True)

# Plot distributions for a subset of features to avoid clutter
features_to_plot = X.columns[:6] # Plotting first 6 features as an example
X_melted = pd.melt(X_scaled, id_vars=['Target'], value_vars=features_to_plot)

plt.figure(figsize=(12, 6))
sns.boxplot(x='variable', y='value', hue='Target', data=X_melted)
plt.title('Normalized Feature Distributions by Class')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
X_scaled = X_scaled.drop('Target', axis=1)

# 3. Split dataset 90/10 and train a classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Using Random Forest
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

train_acc = accuracy_score(y_train, clf.predict(X_train))
test_acc = accuracy_score(y_test, clf.predict(X_test))

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Testing Accuracy: {test_acc:.4f}")

# 4. Change random seed and evaluate thoroughly
X_train_seed, X_test_seed, y_train_seed, y_test_seed = train_test_split(X, y, test_size=0.10, random_state=99)
clf_seed = RandomForestClassifier(random_state=42)
clf_seed.fit(X_train_seed, y_train_seed)
print(f"Testing Accuracy with seed 99: {accuracy_score(y_test_seed, clf_seed.predict(X_test_seed)):.4f}")

# Thorough evaluation using 10-fold cross-validation
cv_scores = cross_val_score(RandomForestClassifier(random_state=42), X, y, cv=10)
print(f"10-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 5. Compute normalized confusion matrix
y_pred = clf.predict(X_test)
cm_normalized = confusion_matrix(y_test, y_pred, normalize='true')

plt.figure(figsize=(6, 5))
sns.heatmap(cm_normalized, annot=True, cmap='Blues', xticklabels=['Dropout (0)', 'Success (1)'], yticklabels=['Dropout (0)', 'Success (1)'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Normalized Confusion Matrix')
plt.show()

# 6. Retrain with 1% of the training set
# 1% of 90% is 0.9% of the total data
X_train_1pct, _, y_train_1pct, _ = train_test_split(X_train, y_train, train_size=0.01, random_state=42, stratify=y_train)

clf_1pct = RandomForestClassifier(random_state=42)
clf_1pct.fit(X_train_1pct, y_train_1pct)
test_acc_1pct = accuracy_score(y_test, clf_1pct.predict(X_test))

print(f"Testing Accuracy (trained on 1% data): {test_acc_1pct:.4f}")