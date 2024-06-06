import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Load the digits dataset
digits = datasets.load_digits()

# Prepare the data
data = digits.images.reshape((len(digits.images), -1))
labels = digits.target

# Normalize the data to match the application preprocessing
data = data / 16.0
data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.5, random_state=42)

# Create a pipeline with data scaling and SVM
pipe = Pipeline([
    ('scaler', StandardScaler()),  # Scale data to zero mean and unit variance
    ('svm', SVC())
])

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': [0.001, 0.01, 0.1, 1],
    'svm__kernel': ['linear', 'rbf']
}
grid = GridSearchCV(pipe, param_grid, refit=True, verbose=2, cv=3)
grid.fit(X_train, y_train)

# Print the best parameters found by GridSearchCV
print(f'Best parameters found: {grid.best_params_}')

# Train the final model with best parameters
clf = grid.best_estimator_
clf.fit(X_train, y_train)

# Test the classifier
predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the model
joblib.dump(clf, 'mnist_svm.pkl')
