

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score

from category_encoders.target_encoder import TargetEncoder

import BaseballDataVisualization1D as bbData

# Load and preview data
df = bbData.csv_to_df()
print(f"\nInitial dataframe shape: {df.shape}")

# Target and feature prep
y = df['description']
X = df.drop(columns=[
'description','game_date','player_name', 'events','game_year','bb_type','spin_dir',
    'spin_rate_deprecated', 'break_angle_deprecated','des','game_type','hit_location',
    'break_length_deprecated', 'tfs_deprecated', 'tfs_zulu_deprecated', 'umpire',
    'sv_id','hc_x','hc_y','hit_distance_sc','launch_speed','launch_angle','game_pk',
    'estimated_ba_using_speedangle','estimated_woba_using_speedangle','woba_value',
    'woba_denom','babip_value','iso_value','launch_speed_angle','post_away_score',
    'post_home_score','post_bat_score','post_fld_score','delta_home_win_exp',
    'delta_run_exp','estimated_slg_using_speedangle','delta_pitcher_run_exp',
    'hyper_speed','home_score_diff','bat_score_diff','home_win_exp',
    'bat_win_exp','age_pit_legacy','age_bat_legacy','n_thruorder_pitcher','type'
])

print(f"\nFeatures remaining after drop: {X.columns.tolist()}")

# Identify column types
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
high_card_cols = [col for col in categorical_cols if X[col].nunique() > 30]
low_card_cols = [col for col in categorical_cols if X[col].nunique() <= 30]
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

print(f"\nCategorical columns: {categorical_cols}")
print(f"→ High-cardinality categorical: {high_card_cols}")
print(f"→ Low-cardinality categorical: {low_card_cols}")
print(f"Numeric columns: {numeric_cols}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")

# Pipelines
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
print("✔ Numeric pipeline created.")

low_card_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
])
print("✔ Low-cardinality categorical pipeline created (OneHot).")

high_card_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('targetenc', TargetEncoder())
])
print("✔ High-cardinality categorical pipeline created (TargetEncoding).")

# Combine transformers
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_cols),
    ('low_card', low_card_pipeline, low_card_cols),
    ('high_card', high_card_pipeline, high_card_cols)
])
print("✔ Preprocessor (ColumnTransformer) assembled.")

# Full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1))
])
print("✔ Full pipeline constructed.")

# Fit model
print("\n🚀 Fitting model...")
pipeline.fit(X_train, y_train)
print("✅ Model training complete.")

# Predict
y_pred = pipeline.predict(X_test)
print("\n📊 Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division='warn'))

# OPTIONAL: Extract encoded feature names
print("\n🔍 Feature transformation summary:")
preprocessor.fit(X_train)

# Get feature names
feature_names = []

# Numeric features
feature_names.extend(numeric_cols)

# OneHot feature names
if low_card_cols:
    onehot = preprocessor.named_transformers_['low_card'].named_steps['onehot']
    onehot_names = onehot.get_feature_names_out(low_card_cols)
    feature_names.extend(onehot_names)

# High-cardinality (TargetEncoded): use original column names
feature_names.extend(high_card_cols)

print(f"Total transformed features: {len(feature_names)}")
print(f"Sample feature names:\n{feature_names[:10]}")

