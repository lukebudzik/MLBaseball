import BaseballDataVisualization1D as bbData
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, precision_score
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = bbData.csv_to_df()

# Drop leakage or irrelevant columns
columnsToRemove = ['description','game_date','player_name', 'events','game_year','bb_type','spin_dir',
    'spin_rate_deprecated', 'break_angle_deprecated','des','game_type','hit_location',
    'break_length_deprecated', 'tfs_deprecated', 'tfs_zulu_deprecated', 'umpire',
    'sv_id','hc_x','hc_y','hit_distance_sc','launch_speed','launch_angle','game_pk',
    'estimated_ba_using_speedangle','estimated_woba_using_speedangle','woba_value',
    'woba_denom','babip_value','iso_value','launch_speed_angle','post_away_score',
    'post_home_score','post_bat_score','post_fld_score','delta_home_win_exp',
    'delta_run_exp','estimated_slg_using_speedangle','delta_pitcher_run_exp',
    'hyper_speed','home_score_diff','bat_score_diff','home_win_exp',
    'bat_win_exp','age_pit_legacy','age_bat_legacy','n_thruorder_pitcher','type']

X = df.drop(columns=columnsToRemove)
y = df['description']

print(f'Input features: {X.columns.tolist()}')

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify column types
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

for col in categorical_cols:
    print(f"{col}: {X[col].nunique()}")

# Filter out high-cardinality categorical columns
categorical_cols = [col for col in categorical_cols if X[col].nunique() < 50]

# Preprocessing with imputers and scaling
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numeric_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# Full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=1))
])

# Train model
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division='warn'))

# Extract feature names
pipeline.named_steps['preprocessor'].fit(X_train)
cat_features = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['ordinal'].get_feature_names_out(categorical_cols)
all_features = np.concatenate([numeric_cols, cat_features])

# Extract and plot feature importances
importances = pipeline.named_steps['classifier'].feature_importances_
indices = np.argsort(importances)[-20:]  # Top 20 features

plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [all_features[i] for i in indices])
plt.xlabel("Feature Importance")
plt.title("Top 20 Important Features (Random Forest)")
plt.tight_layout()
plt.show()



param_dist = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=10, cv=3, n_jobs=1, verbose=1)
X_sample = X_train.sample(frac=0.1, random_state=42)
y_sample = y_train.loc[X_sample.index]
search.fit(X_sample, y_sample)

print("Best parameters:", search.best_params_)
