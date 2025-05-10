import BaseballDataVisualization1D as bbdata
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold


def print_feature_dtypes(df):
    print("\n🧾 Feature Data Types:")
    for col, dtype in df.dtypes.items():
        print(f"  {col:<35} → {dtype}")

def print_dtype_counts(df):
    print("\n🧾 Data type counts:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count}")

def group_features_by_dtype(df):
    print("\n📦 Features Grouped by Data Type:\n")
    dtype_groups = df.dtypes.groupby(df.dtypes).groups  # Maps dtype → list of columns

    for dtype, cols in dtype_groups.items():
        print(f"🔹 {dtype} ({len(cols)} features):")
        for col in sorted(cols):
            print(f"  - {col}")
        print()


def classify_features(df, ordinal_cols=None):
    if ordinal_cols is None:
        ordinal_cols = []

    # Detect potential encoded ordinal features
    potential_encoded_ordinal = [
        col for col in df.select_dtypes(include=['int64', 'int32']).columns
        if 2 <= df[col].nunique() <= 100
    ]
    print("\n🔎 Potential Ordinal Encoded Numericals (low unique values):")
    for col in potential_encoded_ordinal:
        print(f"  {col}: {sorted(df[col].unique().tolist())}")

    # Get column types
    categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Updated discrete vs continuous logic
    discrete = [col for col in numerical if pd.api.types.is_integer_dtype(df[col])]
    continuous = [col for col in numerical if pd.api.types.is_float_dtype(df[col])]

    # Ordinal vs nominal categorical
    ordinal = [col for col in categorical if col in ordinal_cols]
    nominal = [col for col in categorical if col not in ordinal_cols]

    feature_types = {
        'numerical': {
            'discrete': discrete,
            'continuous': continuous
        },
        'categorical': {
            'ordinal': ordinal,
            'nominal': nominal
        }
    }

    # Pretty-print summary
    print("\n📊 Feature Type Summary")
    print(f"Total features: {df.shape[1]}")

    print("\n🧮 Numerical Features")
    print("  ➤ Discrete:")
    print("    " + ", ".join(discrete) if discrete else "    (none)")
    print("  ➤ Continuous:")
    print("    " + ", ".join(continuous) if continuous else "    (none)")

    print("\n🔤 Categorical Features")
    print("  ➤ Ordinal:")
    print("    " + ", ".join(ordinal) if ordinal else "    (none)")
    print("  ➤ Nominal:")
    print("    " + ", ".join(nominal) if nominal else "    (none)")

    return feature_types



if __name__ == '__main__':
    df = bbdata.csv_to_df()
    group_features_by_dtype(df)
    classify_features(df=df)

    print(df['description'].value_counts())

    ##SECOND ITERATION OF ADDITIONS: MORE THOROUGH ANALYSIS OF FEATURES
    '''##Information on sets of features before proceeding with analysis/modifications:
    Input Feature sets:
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
        'batter', 'pitcher', 'on_3b', 'on_2b', 'on_1b', 'fielder_2',
        'fielder_3', 'fielder_4', 'fielder_5', 'fielder_6', 'fielder_7', 'fielder_8',
        'fielder_9', 'stand', 'p_throws', 'inning_topbot', 'pitch_type',
        'home_team', 'away_team', 'zone', 'pitch_name', 'if_fielding_alignment',
        'of_fielding_alignment'
    ]
    categorical_nominal_oneHot = [
        'stand', 'p_throws', 'inning_topbot', 'pitch_type',
        'home_team', 'away_team', 'zone', 'pitch_name', 'if_fielding_alignment',
        'of_fielding_alignment'
    ]
    categorical_nominal_IDs = [
        'batter', 'pitcher', 'on_3b', 'on_2b', 'on_1b', 'fielder_2', 'fielder_3',
        'fielder_4', 'fielder_5', 'fielder_6', 'fielder_7', 'fielder_8', 'fielder_9'
    ]
    Dropped Features and Groupings (excluding the target feature of this iteration, 'description'):
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
    
    '''


