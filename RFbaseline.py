from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from BaseballDataVisualization1D import csv_to_df


def run_random_forest_classification(X, y, random_state=42, n_estimators=100, max_depth=None):
    """
    Trains a Random Forest classifier on the provided data and prints the classification report.
    Automatically handles categorical features via OneHotEncoding.
    """
    # Identify categorical and numerical columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Define preprocessing for columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )

    # Create full pipeline
    pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight='balanced',
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        ))
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )

    # Fit and predict
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))

    return pipeline, y_pred




if __name__ == '__main__':
    df = csv_to_df()
    X = df[['release_speed','release_pos_x','release_pos_z','pfx_x','pfx_z','plate_x','plate_z',
            'vx0','vy0','vz0','ax','ay','az','sz_top','sz_bot','effective_speed','release_spin_rate',
            'release_extension','release_pos_y','bat_speed','swing_length','api_break_z_with_gravity',
            'api_break_x_arm','api_break_x_batter_in','arm_angle','at_bat_number','inning',
            'outs_when_up','balls','strikes','pitch_number','home_score','away_score','bat_score',
            'fld_score','spin_axis','age_pit','age_bat','n_priorpa_thisgame_player_at_bat',
            'pitcher_days_since_prev_game','batter_days_since_prev_game',
            'pitcher_days_until_next_game','batter_days_until_next_game','stand','p_throws',
            'inning_topbot','pitch_type','home_team','away_team','zone','pitch_name',
            'if_fielding_alignment','of_fielding_alignment','batter','pitcher','on_3b','on_2b',
            'on_1b','fielder_2','fielder_3','fielder_4','fielder_5','fielder_6','fielder_7',
            'fielder_8','fielder_9']
    ]
    y= df['description']
    run_random_forest_classification(X,y)