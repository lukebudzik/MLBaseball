import pandas as pd
import numpy as np

from BaseballDataVisualization1D import csv_to_df
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
target = ['description']

def sort_columns_by_dtype(df):
    dtype_groups = {}

    for col in df.columns:
        dtype = df[col].dtype
        dtype_groups.setdefault(dtype, []).append(col)

    print("📊 Columns grouped by data type:")
    for dtype, cols in dtype_groups.items():
        print(f"\n➡️ {dtype} ({len(cols)} columns):")
        for col in cols:
            print(f"   - {col}")

def categorize_columns(df,nan_threshold=0.2,discrete_unique_thresh=15,exclude_substrings=None):
    if exclude_substrings is None:
        exclude_substrings = []

    object_cols = []
    discrete_cols = []
    continuous_cols = []
    nan_rich_cols = []
    excluded_by_name = []

    for col in df.columns:
        # Check for substring exclusion
        if any(substr in col for substr in exclude_substrings):
            excluded_by_name.append(col)
            continue

        # Check NaN ratio
        num_missing = df[col].isna().mean()
        if num_missing > nan_threshold:
            nan_rich_cols.append(col)
            continue

        # Type-based categorization
        col_dtype = df[col].dtype

        if col_dtype == 'object':
            object_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            n_unique = df[col].nunique(dropna=True)
            if n_unique <= discrete_unique_thresh:
                discrete_cols.append(col)
            else:
                continuous_cols.append(col)
#
#     # Print summary
#     print("🚫 Excluded columns by name substring:", len(excluded_by_name))
#     print(excluded_by_name)
#
#     print("\n🧹 Excluded columns with significant NaNs (> {}%):".format(int(nan_threshold * 100)), len(nan_rich_cols))
#     print(nan_rich_cols)
#
#     print("\n🔤 Object columns:", len(object_cols))
#     print(object_cols)
#
#     print("\n🔢 Discrete numeric columns:", len(discrete_cols))
#     print(discrete_cols)
#
#     print("\n📈 Continuous numeric columns:", len(continuous_cols))
#     print(continuous_cols)
#
    return excluded_by_name + nan_rich_cols,object_cols,discrete_cols,continuous_cols

##########################

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freq_maps = {}
        self.columns = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.columns = X.columns
        else:
            raise ValueError("FrequencyEncoder requires a DataFrame input for fitting.")

        self.freq_maps = {col: X[col].value_counts().to_dict() for col in self.columns}
        return self

    def transform(self, X):
        # If it's a DataFrame, use column names directly
        if isinstance(X, pd.DataFrame):
            return X.apply(lambda col: col.map(self.freq_maps.get(col.name, {})).fillna(0))
        # If it's a NumPy array, use self.columns
        elif isinstance(X, np.ndarray):
            if self.columns is None:
                raise ValueError("No column names stored from fitting.")
            df = pd.DataFrame(X, columns=self.columns)
            return df.apply(lambda col: col.map(self.freq_maps.get(col.name, {})).fillna(0)).values
        else:
            raise TypeError("Input must be a DataFrame or a NumPy array.")

class IDPresenceEncoder(BaseEstimator, TransformerMixin):
    """Converts NaN -> 0, non-NaN -> 1"""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_bin = X.notna().astype(int)
        return X_bin

from sklearn.preprocessing import FunctionTransformer

def make_dataframe_transformer(columns):
    return FunctionTransformer(lambda X: pd.DataFrame(X, columns=columns), validate=False)

def build_feature_groups(data):
    excluded, objects2, discrete, continuous = categorize_columns(
        data,
        exclude_substrings=['exp', 'post', 'delta', 'diff', 'legacy']
    )
    excluded += ['game_year', 'game_type', 'des', 'player_name', 'type']
    basesloaded = ['on_3b', 'on_2b', 'on_1b']
    excluded = [item for item in excluded if item not in basesloaded]

    categorical_objects = [
        'pitch_type', 'stand','zone', 'p_throws', 'home_team', 'away_team',
        'inning_topbot', 'pitch_name', 'if_fielding_alignment', 'of_fielding_alignment'
    ]

    identifying_objects = [
        'game_date','fielder_2','fielder_3', 'fielder_4', 'fielder_5',
        'fielder_6', 'fielder_7','fielder_8', 'fielder_9',
        'batter', 'pitcher','game_pk'
    ]

    id_presence_cols = ['on_1b', 'on_2b', 'on_3b']

    numeric_discrete_features = [
        'balls', 'strikes', 'outs_when_up', 'inning', 'n_thruorder_pitcher',
        'n_priorpa_thisgame_player_at_bat','at_bat_number', 'pitch_number', 'home_score',
        'away_score', 'bat_score','fld_score', 'spin_axis', 'age_pit', 'age_bat',
        'pitcher_days_since_prev_game','batter_days_since_prev_game',
        'pitcher_days_until_next_game','batter_days_until_next_game'
    ]

    numeric_continuous_features = [
        'release_speed', 'release_pos_x', 'release_pos_z', 'pfx_x', 'pfx_z',
        'plate_x', 'plate_z', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
        'sz_top', 'sz_bot', 'effective_speed', 'release_spin_rate',
        'release_extension','release_pos_y', 'api_break_z_with_gravity',
        'api_break_x_arm', 'api_break_x_batter_in', 'arm_angle'
    ]

    nonexcluded = categorical_objects + identifying_objects + id_presence_cols + \
                  numeric_discrete_features + numeric_continuous_features

    return {
        'excluded': excluded,
        'categorical': categorical_objects,
        'identifying': identifying_objects,
        'id_flags': id_presence_cols,
        'discrete': numeric_discrete_features,
        'continuous': numeric_continuous_features,
        'nonexcluded': nonexcluded
    }

def print_feature_summary(groups, target):
    print("✅ Included Features:", len(groups['nonexcluded']))
    print(groups['nonexcluded'])

    print("\n🚫 Excluded Features:", len(groups['excluded']))
    print(groups['excluded'])

    common = set(groups['nonexcluded']) & set(groups['excluded'])
    if common:
        print("\n⚠️ Warning: Overlap between excluded and included features!", list(common))

    print("\n🎯 Target:", target)


def print_top_feature_importances(model, preprocessor, top_n=15):
    importances = model.named_steps['clf'].feature_importances_
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: -x[1])[:top_n]
    print("\n🔍 Top Feature Importances:")
    for name, score in feat_imp:
        print(f"{name}: {score:.4f}")
from sklearn.preprocessing import LabelEncoder

def split_feature_types(df, target_column=None, id_columns=None, cardinality_threshold=30):
    """
    Automatically splits DataFrame columns into categorical and numerical features.

    Parameters:
    - df: pandas DataFrame
    - target_column: str or None, column to exclude (e.g. the label)
    - id_columns: list of str, optional columns to exclude (e.g. player_id)
    - cardinality_threshold: int, max unique values for object columns to be treated as categorical

    Returns:
    - categorical_cols: list of column names
    - numerical_cols: list of column names
    """

    if id_columns is None:
        id_columns = []

    exclude = set([target_column] if target_column else []) | set(id_columns)

    categorical_cols = []
    numerical_cols = []

    for col in df.columns:
        if col in exclude:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            numerical_cols.append(col)
        elif pd.api.types.is_object_dtype(df[col]) or isinstance(df[col].dtype, pd.CategoricalDtype):
            if df[col].nunique(dropna=False) <= cardinality_threshold:
                categorical_cols.append(col)
            else:
                # likely an ID or high-cardinality feature, skip or log if needed
                pass

    return categorical_cols, numerical_cols

if __name__ == "__main__":
    data = csv_to_df()
    features = build_feature_groups(data)
    print_feature_summary(features, target)

    y = data[target[0]]
    X = data[features['nonexcluded']]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    id_presence_transformer = Pipeline(steps=[
        ('presence_encoder', IDPresenceEncoder()),  # Step 1: 1 if not NaN
        ('onehot', OneHotEncoder(sparse_output=False, drop=None))  # Step 2: OneHot
    ])

    discrete_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('scaler', StandardScaler())
    ])

    continuous_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    freq_encoder_pipeline = Pipeline([
        ('to_df', make_dataframe_transformer(features['identifying'])),
        ('freq', FrequencyEncoder())
    ])

    model_type = 'rf'

    if model_type == 'rf':
        model = RandomForestClassifier(class_weight= 'balanced',random_state=42)
    elif model_type == 'xgboost':
        from xgboost import XGBClassifier
        model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    elif model_type == 'lightgbm':
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(random_state=42)
    elif model_type == 'catboost':
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(verbose=True, random_seed=42)
    elif model_type == 'mlp':
        model = MLPClassifier()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print("fitting pipeline to ", model_type)


    if model_type == 'mlp':

        categorical_cols, numerical_cols = split_feature_types(
            X,
            target_column='description',
            id_columns=features['identifying']
        )

        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        # Categorical pipeline
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Column transformer
        preprocessor = ColumnTransformer([
            ('num', numeric_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ])

        # === Define MLP model ===
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32),  # Two hidden layers
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )

        # === Build pipeline ===
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # === Train/test split and fit ===
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)

        # === Evaluate ===
        y_pred = pipeline.predict(X_test)
        print(classification_report(y_test, y_pred))
    else:
        pipeline = Pipeline([
            ('preprocessor', ColumnTransformer([
                ('cat', categorical_transformer, features['categorical']),
                ('id', freq_encoder_pipeline, features['identifying']),
                ('disc', discrete_transformer, features['discrete']),
                ('cont', continuous_transformer, features['continuous']),
                ('id_flags', id_presence_transformer, features['id_flags'])
            ], remainder='drop')),
            ('clf', model)
        ])

        pipeline.fit(X_train, np.ravel(y_train))
        y_pred = pipeline.predict(X_test)

        y_pred_labels = label_encoder.inverse_transform(y_pred)
        y_test_labels = label_encoder.inverse_transform(y_test)


        print("\n📊 Classification Report:\n", classification_report(y_test_labels, y_pred_labels))

    #print_top_feature_importances(pipeline, pipeline.named_steps['preprocessor'])


