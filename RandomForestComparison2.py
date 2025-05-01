from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import BaseballDataVisualization1D as bbData
import numpy as np
import pandas as pd

df = bbData.csv_to_df()  # or your preprocessed DataFrame
target_col = 'description'

# Replace with your actual feature sets
remove_features = ['description','game_date','player_name', 'events','game_year','bb_type','spin_dir',
               'spin_rate_deprecated', 'break_angle_deprecated','des','game_type','hit_location',
               'break_length_deprecated', 'tfs_deprecated', 'tfs_zulu_deprecated', 'umpire',
               'sv_id','hc_x','hc_y','hit_distance_sc','launch_speed','launch_angle','game_pk',
               'estimated_ba_using_speedangle','estimated_woba_using_speedangle','woba_value',
               'woba_denom','babip_value','iso_value','launch_speed_angle','post_away_score',
               'post_home_score','post_bat_score','post_fld_score','delta_home_win_exp',
               'delta_run_exp','estimated_slg_using_speedangle','delta_pitcher_run_exp',
               'hyper_speed','home_score_diff','bat_score_diff','home_win_exp',
               'bat_win_exp','age_pit_legacy','age_bat_legacy','n_thruorder_pitcher']
print(f'number of removed features:', len(remove_features))

original_feature_list = [col for col in df.columns if col != target_col]
cleaned_feature_list = [col for col in original_feature_list if
                        col not in remove_features]

print(f'number of original features:', len(original_feature_list))
print(f'number of features after removal:', len(cleaned_feature_list))



X_with = df[original_feature_list]
X_without = df[cleaned_feature_list]
y = df[target_col]


sample_frac = 0.25  # Use 25% of data
X_with_sample = X_with.sample(frac=sample_frac, random_state=42)
X_without_sample = X_without.loc[X_with_sample.index]
y_sample = y.loc[X_with_sample.index]

def build_pipeline(X):
    # Detect types
    categorical_cols = X[['batter','pitcher','stand','p_throws', 'inning_topbot', 'pitch_type',
    'home_team','away_team','type','on_3b','on_2b','on_1b','fielder_2','fielder_3','fielder_4',
    'fielder_5','fielder_6','fielder_7','fielder_8','fielder_9','zone','pitch_name',
    'if_fielding_alignment', 'of_fielding_alignment']].columns.tolist()
    numeric_cols = X[['release_speed','release_pos_x','release_pos_z','pfx_x', 'pfx_z', 'plate_x',
    'plate_z','vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'sz_top','sz_bot', 'effective_speed',
    'release_spin_rate', 'release_extension','release_pos_y','bat_speed','swing_length',
    'api_break_z_with_gravity','api_break_x_arm', 'api_break_x_batter_in', 'arm_angle']].columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    return pipeline, preprocessor, numeric_cols, categorical_cols

def get_feature_names(preprocessor, numeric_cols, categorical_cols, X):
    # Get numeric feature names
    num_features = numeric_cols

    # Get categorical feature names after one-hot encoding
    cat_transformer = preprocessor.named_transformers_['cat']
    ohe = cat_transformer.named_steps['onehot']
    cat_features = ohe.get_feature_names_out(categorical_cols)

    # Combine
    return np.concatenate([num_features, cat_features])

# -----------------------------
# 4. Evaluate with cross-validation and fit for importances
# -----------------------------
pipeline_with, preprocessor_with, num_cols_with, cat_cols_with = build_pipeline(X_with_sample)
pipeline_without, preprocessor_without, num_cols_without, cat_cols_without = build_pipeline(X_without_sample)

score_with = cross_val_score(pipeline_with, X_with_sample, y_sample, cv=3, n_jobs=-1).mean()
score_without = cross_val_score(pipeline_without, X_without_sample, y_sample, cv=3, n_jobs=-1).mean()

# Fit pipelines to get feature importances
pipeline_with.fit(X_with_sample, y_sample)
pipeline_without.fit(X_without_sample, y_sample)

# Get feature names
feature_names_with = get_feature_names(
    pipeline_with.named_steps['preprocessor'],
    num_cols_with,
    cat_cols_with,
    X_with_sample
)
feature_names_without = get_feature_names(
    pipeline_without.named_steps['preprocessor'],
    num_cols_without,
    cat_cols_without,
    X_without_sample
)

# Get importances
importances_with = pipeline_with.named_steps['classifier'].feature_importances_
importances_without = pipeline_without.named_steps['classifier'].feature_importances_

# Display top 20 features for each model
def display_top_features(feature_names, importances, title):
    indices = np.argsort(importances)[::-1][:20]
    print(f"\nTop 20 Feature Importances for {title}:")
    for idx in indices:
        print(f"{feature_names[idx]}: {importances[idx]:.4f}")

# -----------------------------
# 5. Print Results
# -----------------------------
print(f"Model with all features:       {score_with:.4f}")
print(f"Model without leakage features: {score_without:.4f}")

display_top_features(feature_names_with, importances_with, "ALL FEATURES")
display_top_features(feature_names_without, importances_without, "NO LEAKAGE FEATURES")
