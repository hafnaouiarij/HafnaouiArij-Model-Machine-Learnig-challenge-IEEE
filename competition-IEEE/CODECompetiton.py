import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import randint

## Load data
data = pd.read_csv('data/train.csv')

# Feature Engineering
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['Watermelon Disease Presence'] = data['Watermelon Disease Presence'].map({'Yes': 1, 'No': 0})
data['Watermelon Quality (Category)'] = data['Watermelon Quality (Category)'].map({'Below Standard': 1, 'Standard': 2, 'Premium': 3})
data['Pest/Disease Incidence'] = data['Pest/Disease Incidence'].map({'Low': 1, 'Medium': 2, 'High': 3})
data.drop(['Date', 'Watermelon Variety', 'Geographical Location', 'Harvest Time'], axis=1, inplace=True)
data.fillna(data.mean(), inplace=True)
data = pd.get_dummies(data)

# Split data into features and target
X = data.drop('Watermelon Disease Presence', axis=1)
y = data['Watermelon Disease Presence']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter distributions to sample from
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [None] + list(range(5, 50)),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'max_features': ['sqrt', 'log2'] + list(range(1, X.shape[1])),
    'bootstrap': [True, False]
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_distributions=param_dist, n_iter=100, cv=3, scoring='f1', random_state=42, n_jobs=-1, verbose=2)

# Perform Randomized Search
random_search.fit(X_train, y_train)

# Get the best parameters
best_params = random_search.best_params_

# Train the model with the best parameters
best_model = RandomForestClassifier(random_state=42, **best_params)
best_model.fit(X_train, y_train)

# Make predictions on the test data
best_predictions = best_model.predict(X_test)

# Calculate evaluation metrics
best_f_score = f1_score(y_test, best_predictions)
conf_matrix = confusion_matrix(y_test, best_predictions)
accuracy = accuracy_score(y_test, best_predictions)
precision = precision_score(y_test, best_predictions)
recall = recall_score(y_test, best_predictions)

print("Best F-score after hyperparameter tuning:", best_f_score)
print("Best parameters:", best_params)
print("Confusion Matrix:")
print(conf_matrix)
print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F-score:", best_f_score)

# Load test data
data2 = pd.read_csv('data/test.csv')

# Feature Engineering for test data
data2['Date'] = pd.to_datetime(data2['Date'])
data2['Year'] = data2['Date'].dt.year
data2['Month'] = data2['Date'].dt.month
data2['Day'] = data2['Date'].dt.day
data2['Watermelon Quality (Category)'] = data2['Watermelon Quality (Category)'].map({'Below Standard': 1, 'Standard': 2, 'Premium': 3})
data2['Pest/Disease Incidence'] = data2['Pest/Disease Incidence'].map({'Low': 1, 'Medium': 2, 'High': 3})
data2.drop(['Date', 'Watermelon Variety', 'Geographical Location', 'Harvest Time'], axis=1, inplace=True)
data2.fillna(data2.mean(), inplace=True)
data2 = pd.get_dummies(data2)

# Make predictions using the best model
test_predictions = best_model.predict(data2)

# Create a DataFrame with 'ID' and 'Watermelon Disease Presence' columns
results_df = pd.DataFrame({data2['ID'], 'Watermelon Disease Presence'})
# Map 1 to 'Yes' and 0 to 'No'
results_df['Watermelon Disease Presence'] = results_df['Watermelon Disease Presence'].map({1: 'Yes', 0: 'No'})

# Display the DataFrame
print(results_df)