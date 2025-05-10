import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler
)
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report

from BaseballDataVisualization1D import csv_to_df

X = csv_to_df()

# --- ✳️ Define feature groups (customize as needed) ---
categorical_features = ['pitch_type','zone','home_team','away_team','if_fielding_alignment','of_fielding_alignment']
id_features = ['batter','pitcher','game_pk','fielder_2','fielder_3','fielder_4','fielder_5',
               'fielder_6','fielder_7','fielder_8','fielder_9']
continuous_features = ['release_speed', 'release_pos_x', 'release_pos_z', 'pfx_x', 'pfx_z',
                        'plate_x', 'plate_z', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
                        'sz_top', 'sz_bot', 'effective_speed', 'release_spin_rate',
                        'release_extension', 'release_pos_y', 'api_break_z_with_gravity',
                        'api_break_x_arm', 'api_break_x_batter_in', 'arm_angle']
discrete_features = ['outs_when_up', 'inning', 'balls', 'strikes', 'at_bat_number',
                    'pitch_number', 'home_score', 'away_score', 'bat_score', 'fld_score',
                    'age_pit', 'age_bat', 'n_thruorder_pitcher', 'n_priorpa_thisgame_player_at_bat',
                    'pitcher_days_since_prev_game', 'batter_days_since_prev_game',
                    'pitcher_days_until_next_game', 'batter_days_until_next_game', 'spin_axis']
binary_features = ['stand','p_throws','inning_topbot']# 'on_#b' encoded 1 if not null
flag_features = ['on_3b','on_2b','on_1b']
inputfeatures= (categorical_features + id_features + continuous_features + discrete_features +
                binary_features + flag_features)
targetFeature = ['description']

deprecated = ['spin_dir', 'spin_rate_deprecated', 'break_angle_deprecated', 'break_length_deprecated',
              'des', 'hc_x', 'hc_y', 'tfs_deprecated', 'tfs_zulu_deprecated', 'umpire',
             'sv_id', 'age_pit_legacy', 'age_bat_legacy']
redundant = ['player_name','game_date','game_type','game_year','pitch_name']
leakage = ['events','hit_distance_sc', 'launch_speed', 'launch_angle', 'estimated_ba_using_speedangle',
            'estimated_woba_using_speedangle', 'woba_value', 'woba_denom', 'babip_value',
            'iso_value', 'launch_speed_angle', 'delta_home_win_exp', 'delta_run_exp',
            'bat_speed', 'swing_length', 'estimated_slg_using_speedangle',
           'delta_pitcher_run_exp', 'hyper_speed', 'post_away_score', 'post_home_score',
          'post_bat_score', 'post_fld_score', 'home_score_diff', 'bat_score_diff',
           'home_win_exp', 'bat_win_exp','type','hit_location','bb_type']
excluded_features = deprecated + redundant + leakage + targetFeature

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
])

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freq_maps = {}

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            for col in X.columns:
                self.freq_maps[col] = X[col].value_counts()
        else:
            raise ValueError("FrequencyEncoder requires a pandas DataFrame")
        return self

    def transform(self, X):
        X_encoded = X.copy()
        for col in X.columns:
            X_encoded[col] = X[col].map(self.freq_maps[col]).fillna(0)
        return X_encoded


id_freq_transformer = Pipeline(steps=[
    ('freq', FrequencyEncoder())
])

continuous_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

discrete_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    # DEFINE DISCRETE TRANSFORMATION BASED ON MODEL CHOICE
    ('scaler', StandardScaler())
])

binary_category_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # if needed
    ('encoder', OneHotEncoder(drop='if_binary', sparse_output=False))
])
flag_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0))  # if needed
    # No encoding: these are already numeric flags
])


preprocessor = ColumnTransformer(transformers=[
    ('cat', categorical_transformer, categorical_features),
    ('id', id_freq_transformer, id_features),
    ('cont', continuous_transformer, continuous_features),
    ('disc', discrete_transformer, discrete_features),
    ('bin', binary_category_transformer, binary_features),
    ('flag', flag_transformer, flag_features)
])

param_distributions = {
        'clf__n_estimators': [50, 100, 200],
        'clf__max_depth': [5, 10, None],
        'clf__min_samples_split': [2, 5, 10]
    }
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
     # Add a classifier or regressor here if training:
    ('clf', RandomForestClassifier(random_state=42))
])


if __name__ == '__main__':
    y= X[targetFeature]
    print("features excluded from input space (",len(excluded_features) ,"): ", excluded_features)
    X = X.drop(columns=excluded_features)

    print("input features (",X.shape[1],"):", X.columns.tolist())
    print("target feature(s): ", targetFeature)




    # Split data (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y[targetFeature[0]], test_size=0.2, random_state=42)

    missing_cols = [col for col in (categorical_features + id_features + continuous_features + binary_features + discrete_features + flag_features) if col not in X_train.columns.tolist()]
    print("Missing columns:", missing_cols)


    X_cat = X_train[categorical_features]
    X_cont = X_train[continuous_features]
    X_bin = X_train[binary_features]
    X_ids = X_train[id_features]
    X_disc = X_train[discrete_features]
    X_flag = X_train[flag_features]

    categorical_transformer.fit(X_cat)
    continuous_transformer.fit(X_cont)
    binary_category_transformer.fit(X_bin)
    id_freq_transformer.fit(X_ids)
    discrete_transformer.fit(X_disc)
    flag_transformer.fit(X_flag)

    preprocessor.fit(X_train,y_train)

    # Randomized search
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=5,  # try 5 random combos
        cv=3,  # 3-fold CV
        verbose=1,
        n_jobs=1,  # <= lower memory
        random_state=42
    )

    # Run search
    search.fit(X_train, y_train)

    # Evaluate
    print("Best params:", search.best_params_)
    y_pred = search.predict(X_test)
    print(classification_report(y_test, y_pred))
