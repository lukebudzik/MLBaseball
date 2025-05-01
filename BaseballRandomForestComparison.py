
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import BaseballDataVisualization1D as bbData

from joblib import Memory


# -----------------------------
# 1. Load your data
# -----------------------------
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

# -----------------------------
# 2. Select data for comparison
# -----------------------------
X_with = df[original_feature_list]
X_without = df[cleaned_feature_list]
y = df[target_col]

sample_frac = 0.25  # Use 25% of data
X_with_sample = X_with.sample(frac=sample_frac, random_state=42)
X_without_sample = X_without.loc[X_with_sample.index]
y_sample = y.loc[X_with_sample.index]


# -----------------------------
# 3. Preprocessing setup
# -----------------------------
def build_pipeline(X):
    # Detect types
    print("building pipeline...")
    categorical_cols = X[['batter','pitcher','stand','p_throws', 'inning_topbot', 'pitch_type',
    'home_team','away_team','type','on_3b','on_2b','on_1b','fielder_2','fielder_3','fielder_4',
    'fielder_5','fielder_6','fielder_7','fielder_8','fielder_9','zone','pitch_name',
    'if_fielding_alignment', 'of_fielding_alignment']].columns.tolist()
    numeric_cols = X[['release_speed','release_pos_x','release_pos_z','pfx_x', 'pfx_z', 'plate_x',
    'plate_z','vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'sz_top','sz_bot', 'effective_speed',
    'release_spin_rate', 'release_extension','release_pos_y','bat_speed','swing_length',
    'api_break_z_with_gravity','api_break_x_arm', 'api_break_x_batter_in', 'arm_angle']].columns.tolist()


    # transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])


    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    return pipeline


# -----------------------------
# 4. Evaluate with cross-validation
# -----------------------------
pipeline_with = build_pipeline(X_with_sample)
pipeline_without = build_pipeline(X_without_sample)

score_with = cross_val_score(pipeline_with, X_with_sample, y_sample, cv=3, n_jobs=-1).mean()
score_without = cross_val_score(pipeline_without, X_without_sample, y_sample, cv=3, n_jobs=-1).mean()

# -----------------------------
# 5. Print Results
# -----------------------------
print(f"Model with all features:       {score_with:.4f}")
print(f"Model without leakage features: {score_without:.4f}")

