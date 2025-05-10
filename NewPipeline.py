from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier  # or any model you want
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import BaseballDataVisualization1D as bbData
import numpy as np
from category_encoders import HashingEncoder

# --- Your manually grouped features ---
numerical_continuous = [
    'release_speed', 'release_pos_x', 'release_pos_z', 'pfx_x', 'pfx_z',
    'plate_x', 'plate_z', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'sz_top',
    'sz_bot', 'effective_speed', 'release_spin_rate', 'release_extension',
    'release_pos_y', 'bat_speed', 'swing_length', 'api_break_z_with_gravity',
    'api_break_x_arm', 'api_break_x_batter_in', 'arm_angle'
]

numerical_discrete = [
    'at_bat_number', 'inning', 'outs_when_up', 'balls', 'strikes',
    'pitch_number', 'home_score', 'away_score', 'bat_score', 'fld_score',
    'spin_axis', 'age_pit', 'age_bat', 'n_priorpa_thisgame_player_at_bat',
    'pitcher_days_since_prev_game', 'batter_days_since_prev_game',
    'pitcher_days_until_next_game', 'batter_days_until_next_game'
]

categorical_nominal = [
    'batter', 'pitcher',  'on_3b', 'on_2b', 'on_1b', 'fielder_2',
    'fielder_3', 'fielder_4', 'fielder_5', 'fielder_6', 'fielder_7', 'fielder_8',
    'fielder_9','stand', 'p_throws', 'inning_topbot', 'pitch_type',
    'home_team', 'away_team','zone', 'pitch_name', 'if_fielding_alignment',
    'of_fielding_alignment'
]
categorical_nom_oneHot = [
    'stand', 'p_throws', 'inning_topbot', 'pitch_type',
    'home_team', 'away_team','zone', 'pitch_name', 'if_fielding_alignment',
    'of_fielding_alignment'
]
categorical_nom_Hash = [
    'batter', 'pitcher',  'on_3b', 'on_2b', 'on_1b', 'fielder_2','fielder_3',
    'fielder_4', 'fielder_5', 'fielder_6', 'fielder_7', 'fielder_8','fielder_9'
]

# --- Transformers ---
continuous_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('scale', StandardScaler())
])

discrete_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent'))
    # optionally add scaling or leave raw
])


categorical_pipeline_Hash = Pipeline([
    ('impute', SimpleImputer(strategy='constant')),
    ('hashing', HashingEncoder(n_components=32))  # Try 32, 64, etc.
])


categorical_pipeline_OneHot = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=200))
])

# --- ColumnTransformer combining all ---
preprocessor = ColumnTransformer([
    ('cont', continuous_pipeline, numerical_continuous),
    ('disc', discrete_pipeline, numerical_discrete),
    ('OH', categorical_pipeline_OneHot, categorical_nom_oneHot),
    ('Hash', categorical_pipeline_Hash, categorical_nom_Hash)
])

# --- Final model pipeline ---
model_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', RandomForestClassifier(class_weight='balanced',n_estimators=100, random_state=42))  # or any classifier
])

def get_feature_names(preprocessor):
    output_features = []

    for name, transformer, columns in preprocessor.transformers_:
        if transformer == 'drop' or name == 'remainder':
            continue

        # Handle pipeline
        if hasattr(transformer, 'named_steps'):
            last_step = list(transformer.named_steps.values())[-1]

            if isinstance(last_step, OneHotEncoder):
                feature_names = last_step.get_feature_names_out(columns)
            elif isinstance(last_step, HashingEncoder):
                # HashingEncoder does not retain original names
                n_components = last_step.n_components
                feature_names = [f"{name}_hash_{i}" for i in range(n_components)]
            else:
                feature_names = columns

        # Handle non-pipeline transformers
        elif hasattr(transformer, 'get_feature_names_out'):
            feature_names = transformer.get_feature_names_out(columns)
        else:
            feature_names = columns

        output_features.extend(feature_names)

    return output_features


if __name__ == '__main__':
    df = bbData.csv_to_df()
    df[categorical_nom_Hash] = df[categorical_nom_Hash].astype(str)
    targetFeatures = 'description'
    leakyFeatures = [
        'hit_distance_sc','events','bb_type','hit_location','hc_x','hc_y',
        'launch_speed','launch_angle','estimated_ba_using_speedangle',
        'estimated_woba_using_speedangle','woba_value','woba_denom','babip_value',
        'iso_value','launch_speed_angle','post_away_score','post_home_score',
        'post_bat_score','post_fld_score','delta_home_win_exp','delta_run_exp',
        'estimated_slg_using_speedangle','delta_pitcher_run_exp','hyper_speed',
        'home_score_diff','bat_score_diff','home_win_exp','bat_win_exp','type',
    ]
    deprecatedFeatures = [
        'spin_dir','spin_rate_deprecated', 'break_angle_deprecated','des',
        'break_length_deprecated', 'tfs_deprecated', 'tfs_zulu_deprecated','umpire','sv_id',
        'age_pit_legacy','age_bat_legacy'
    ]

    redundantFeatures = [
        'game_year','player_name','game_type'
    ]
    colsToDrop = (
        ['game_date','game_pk','n_thruorder_pitcher']
        + leakyFeatures
        + deprecatedFeatures
        + redundantFeatures
    ) # also redundant features
    print("dropped features: ", colsToDrop)
    print("# dropped features:", len(colsToDrop))


    df = df.drop(columns=colsToDrop)

    X = df[numerical_continuous + numerical_discrete + categorical_nominal]

    #X = X.sample(frac=.1, random_state=42)
    #y = y
    inputFeatures = X.columns.tolist()
    print("input features: ", inputFeatures)
    print("# input features: ", len(inputFeatures))
    y = df[targetFeatures]  # replace with your actual target
    print("target feature(s): ", targetFeatures)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  # stratify helps with class balance
    )
    print("fitting model...")
    model_pipeline.fit(X_train, np.ravel(y_train))
    print("model fit! making predictions on 20% testing set...")

    y_pred = model_pipeline.predict(X_test)

    print(classification_report(y_test, y_pred))

    print("getting feature importances...")

    preprocessor = model_pipeline.named_steps['preprocess']
    feature_names = get_feature_names(preprocessor)
    importances = model_pipeline.named_steps['model'].feature_importances_

    # Ensure alignment between importances and feature names
    if len(importances) != len(feature_names):
        print(
            f"⚠️ Warning: mismatch between feature importances ({len(importances)}) and feature names ({len(feature_names)})")
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    sorted_indices = np.argsort(importances)[::-1]

    print("\nTop 20 features by importance:\n")
    for idx in sorted_indices[:20]:
        print(f"{feature_names[idx]}: {importances[idx]:.4f}")



