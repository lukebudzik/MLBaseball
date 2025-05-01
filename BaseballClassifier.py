import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the data
data = pd.read_csv('statcast_20172.csv')

# Encode categorical variables
le = LabelEncoder()
categorical_columns = ['player_id', 'pitch_type', 'stand', 'p_throws', 'balls', 'strikes']

for col in categorical_columns:
    data[col] = data[col].fillna("Unknown")  # Handle missing values
    data[col] = data[col].astype(str)  # Convert to string before encoding
    data[col] = le.fit_transform(data[col])  # Apply encoding

# Encode target variable 'type'
le_type = LabelEncoder()
data['type'] = le_type.fit_transform(data['type'])

le_hit_location = LabelEncoder()
# Only encode hit_location where type == 'X'
X_label = 'X'
data.loc[data['type'] == le_type.transform([X_label])[0], 'hit_location'] = (
    le_hit_location.fit_transform(data.loc[data['type'] == le_type.transform([X_label])[0], 'hit_location'].fillna("Unknown").astype(str))
)

# Separate features and target
X = data.drop(['type', 'hit_location'], axis=1, errors='ignore')
y_type = data['type']

# Split the data for type prediction
X_train, X_test, y_type_train, y_type_test = train_test_split(X, y_type, test_size=0.2, random_state=42)

# Train model for type prediction
rf_type = RandomForestClassifier(n_estimators=100, random_state=42)
rf_type.fit(X_train, y_type_train)

# Make predictions for type
y_type_pred = rf_type.predict(X_test)

# Evaluate the type model
type_accuracy = accuracy_score(y_type_test, y_type_pred)
print("Pitch type prediction:")
print(f"Accuracy: {type_accuracy:.2f}")
print(classification_report(y_type_test, y_type_pred))

# Filter data for X outcomes
in_play_data = data[data['type'] == le_type.transform([X_label])[0]]
X_in_play = in_play_data.drop(['type', 'hit_location'], axis=1, errors='ignore')
y_hit_location = in_play_data['hit_location']

# Split the X data for hit location prediction
X_train_hit, X_test_hit, y_train_hit, y_test_hit = train_test_split(X_in_play, y_hit_location, test_size=0.2, random_state=42)

# Train model for hit location
rf_hit = RandomForestClassifier(n_estimators=100, random_state=42)
rf_hit.fit(X_train_hit, y_train_hit.to_numpy())

# Make predictions for hit location
y_pred_hit = rf_hit.predict(X_test_hit)


# Evaluate the hit location model
hit_accuracy = accuracy_score(y_test_hit, y_pred_hit)
print("\nHit location prediction (for in-play outcomes only):")
print(f"Accuracy: {hit_accuracy:.3f}")
print(classification_report(y_test_hit, y_pred_hit))

# Function to predict both type and hit location
def predict_outcome(features):
    type_prediction = rf_type.predict(features)[0]
    predicted_type = le_type.inverse_transform([type_prediction])[0]  # Convert back to category

    if predicted_type == X_label:
        hit_location_prediction = rf_hit.predict(features)[0]
        return predicted_type, le_hit_location.inverse_transform([hit_location_prediction])[0]
    else:
        return predicted_type, None

# Example usage
sample_features = X_test.iloc[[0]]
predicted_type, predicted_location = predict_outcome(sample_features)
print(f"\nPredicted type: {predicted_type}")
print(f"Predicted hit location: {predicted_location}")
