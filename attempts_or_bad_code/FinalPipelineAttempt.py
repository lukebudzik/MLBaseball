import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report
import numpy as np
import warnings

from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

def safe_transform(estimator, X):
    try:
        check_is_fitted(estimator)
        return estimator.transform(X)
    except NotFittedError:
        print(f"[❌ ERROR] {type(estimator).__name__} is not fitted.")
        raise

def get_feature_names(preprocessor):
    output_features = []
    for name, trans, cols in preprocessor.transformers_:
        try:
            if hasattr(trans, 'get_feature_names_out'):
                if hasattr(trans, 'named_steps'):
                    step = trans.named_steps[list(trans.named_steps.keys())[-1]]
                    names = step.get_feature_names_out(cols)
                else:
                    names = trans.get_feature_names_out(cols)
                output_features.extend(names)
            else:
                output_features.extend(cols)
        except NotFittedError:
            print(f"[⚠️ WARNING] Transformer '{name}' not fitted. Skipping.")
            continue
    return output_features


# Import your data loading function
from BaseballDataVisualization1D import csv_to_df


def build_baseball_pipeline():
    """
    Build and train a preprocessing pipeline for Statcast baseball data
    with DataFrame preservation in transformers
    """
    # Load the data
    X = csv_to_df()

    # --- Define feature groups ---
    categorical_features = ['pitch_type', 'zone', 'home_team', 'away_team', 'if_fielding_alignment',
                            'of_fielding_alignment']
    id_features = ['batter', 'pitcher', 'game_pk', 'fielder_2', 'fielder_3', 'fielder_4', 'fielder_5',
                   'fielder_6', 'fielder_7', 'fielder_8', 'fielder_9']
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
    binary_features = ['stand', 'p_throws', 'inning_topbot']
    flag_features = ['on_3b', 'on_2b', 'on_1b']

    targetFeature = 'description'

    deprecated = ['spin_dir', 'spin_rate_deprecated', 'break_angle_deprecated', 'break_length_deprecated',
                  'des', 'hc_x', 'hc_y', 'tfs_deprecated', 'tfs_zulu_deprecated', 'umpire',
                  'sv_id', 'age_pit_legacy', 'age_bat_legacy']
    redundant = ['player_name', 'game_date', 'game_type', 'game_year', 'pitch_name']
    leakage = ['events', 'hit_distance_sc', 'launch_speed', 'launch_angle', 'estimated_ba_using_speedangle',
               'estimated_woba_using_speedangle', 'woba_value', 'woba_denom', 'babip_value',
               'iso_value', 'launch_speed_angle', 'delta_home_win_exp', 'delta_run_exp',
               'bat_speed', 'swing_length', 'estimated_slg_using_speedangle',
               'delta_pitcher_run_exp', 'hyper_speed', 'post_away_score', 'post_home_score',
               'post_bat_score', 'post_fld_score', 'home_score_diff', 'bat_score_diff',
               'home_win_exp', 'bat_win_exp', 'type', 'hit_location', 'bb_type']

    excluded_features = deprecated + redundant + leakage

    # Define data type converter that preserves DataFrame structure
    class DataTypeConverter(BaseEstimator, TransformerMixin):
        """
        Transformer to enforce consistent data types in columns while preserving DataFrame structure
        """

        def __init__(self, categorical_cols=None, numeric_cols=None):
            self.categorical_cols = categorical_cols if categorical_cols is not None else []
            self.numeric_cols = numeric_cols if numeric_cols is not None else []

        def fit(self, X, y=None):
            # Nothing to learn
            return self

        def transform(self, X):
            # Make sure X is a DataFrame
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)

            X_transformed = X.copy()

            # Convert categorical columns to string
            for col in self.categorical_cols:
                if col in X_transformed.columns:
                    X_transformed[col] = X_transformed[col].astype(str)

            # Convert numeric columns to float
            for col in self.numeric_cols:
                if col in X_transformed.columns:
                    # Try to convert to numeric, set errors to NaN
                    X_transformed[col] = pd.to_numeric(X_transformed[col], errors='coerce')

            return X_transformed

    # Improved Simple Imputer that preserves DataFrame structure
    class DataFrameImputer(BaseEstimator, TransformerMixin):
        """
        Wrapper around SimpleImputer that preserves DataFrame structure
        """

        def __init__(self, strategy='mean', fill_value=None):
            self.strategy = strategy
            self.fill_value = fill_value
            self.imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
            self.columns = None

        def fit(self, X, y=None):
            # Store column names
            self.columns = X.columns if isinstance(X, pd.DataFrame) else None
            # Fit the imputer
            self.imputer.fit(X)
            return self

        def transform(self, X):
            # Transform the data
            X_transformed = self.imputer.transform(X)
            # Restore DataFrame structure if it was one
            if self.columns is not None:
                X_transformed = pd.DataFrame(X_transformed, index=X.index if hasattr(X, 'index') else None,
                                             columns=self.columns)
            return X_transformed

    # Define improved FrequencyEncoder that explicitly handles DataFrames
    class FrequencyEncoder(BaseEstimator, TransformerMixin):
        def __init__(self, min_frequency=0.001):
            self.freq_maps = {}
            self.min_frequency = min_frequency
            self.default_value = 0
            self.columns = None

        def fit(self, X, y=None):
            # Ensure X is a DataFrame
            if not isinstance(X, pd.DataFrame):
                raise ValueError("FrequencyEncoder requires a pandas DataFrame")

            self.columns = X.columns

            for col in X.columns:
                # Ensure all values are strings for consistent encoding
                values = X[col].astype(str)
                # Calculate frequencies
                freq = values.value_counts(normalize=True)
                # Filter out rare categories
                self.freq_maps[col] = freq[freq >= self.min_frequency]

            return self

        def transform(self, X):
            # Ensure X is a DataFrame
            if not isinstance(X, pd.DataFrame):
                if hasattr(X, 'shape') and hasattr(X, 'ndim') and X.ndim == 2:
                    # Try to convert array-like to DataFrame
                    X = pd.DataFrame(X, columns=self.columns)
                else:
                    raise ValueError("FrequencyEncoder requires a pandas DataFrame")

            X_encoded = pd.DataFrame(index=X.index)

            for col in X.columns:
                if col in self.freq_maps:
                    # Convert to string first for consistent lookup
                    values = X[col].astype(str)
                    # Map values and handle unseen categories
                    X_encoded[col] = values.map(lambda x: self.freq_maps[col].get(x, self.default_value))
                else:
                    X_encoded[col] = self.default_value

            return X_encoded

        def get_feature_names_out(self, input_features=None):
            """Return feature names for output features."""
            if input_features is None:
                return np.array(self.columns, dtype=object)
            return np.array(input_features, dtype=object)

    # DataFrame-preserving wrapper for OneHotEncoder
    class DataFrameOneHotEncoder(BaseEstimator, TransformerMixin):
        def __init__(self, drop=None, sparse_output=False, handle_unknown='ignore'):
            self.drop = drop
            self.sparse_output = sparse_output
            self.handle_unknown = handle_unknown
            self.encoder = OneHotEncoder(drop=drop, sparse_output=sparse_output, handle_unknown=handle_unknown)
            self.columns = None
            self.feature_names_out_ = None

        def fit(self, X, y=None):
            # Store column names
            self.columns = X.columns if isinstance(X, pd.DataFrame) else None
            # Fit the encoder
            self.encoder.fit(X)
            # Get feature names
            self.feature_names_out_ = self.encoder.get_feature_names_out(self.columns)
            return self

        def transform(self, X):
            # Transform the data
            X_transformed = self.encoder.transform(X)
            # Restore DataFrame structure
            if not self.sparse_output:
                X_transformed = pd.DataFrame(
                    X_transformed,
                    index=X.index if hasattr(X, 'index') else None,
                    columns=self.feature_names_out_
                )
            return X_transformed

        def get_feature_names_out(self, input_features=None):
            return self.feature_names_out_

    # DataFrame-preserving wrapper for StandardScaler
    class DataFrameStandardScaler(BaseEstimator, TransformerMixin):
        def __init__(self):
            self.scaler = StandardScaler()
            self.columns = None

        def fit(self, X, y=None):
            # Store column names
            self.columns = X.columns if isinstance(X, pd.DataFrame) else None
            # Fit the scaler
            self.scaler.fit(X)
            return self

        def transform(self, X):
            # Transform the data
            X_transformed = self.scaler.transform(X)
            # Restore DataFrame structure
            if self.columns is not None:
                X_transformed = pd.DataFrame(
                    X_transformed,
                    index=X.index if hasattr(X, 'index') else None,
                    columns=self.columns
                )
            return X_transformed

    # DataFrame-preserving wrapper for RobustScaler
    class DataFrameRobustScaler(BaseEstimator, TransformerMixin):
        def __init__(self):
            self.scaler = RobustScaler()
            self.columns = None

        def fit(self, X, y=None):
            # Store column names
            self.columns = X.columns if isinstance(X, pd.DataFrame) else None
            # Fit the scaler
            self.scaler.fit(X)
            return self

        def transform(self, X):
            # Transform the data
            X_transformed = self.scaler.transform(X)
            # Restore DataFrame structure
            if self.columns is not None:
                X_transformed = pd.DataFrame(
                    X_transformed,
                    index=X.index if hasattr(X, 'index') else None,
                    columns=self.columns
                )
            return X_transformed

    # Remove excluded features
    exclude_cols = [col for col in excluded_features if col in X.columns]
    X = X.drop(columns=exclude_cols)

    # Extract target
    if targetFeature not in X.columns:
        raise ValueError(f"Target feature '{targetFeature}' not found in dataset")

    y = X[targetFeature]
    X = X.drop(columns=[targetFeature])

    # Ensure target is string type for classification
    y = y.astype(str)

    # Check for mixed data types in columns
    mixed_type_cols = []
    for col in X.columns:
        if X[col].dropna().apply(type).nunique() > 1:
            mixed_type_cols.append(col)

    if mixed_type_cols:
        print(f"\nWarning: {len(mixed_type_cols)} columns have mixed data types:")
        for col in mixed_type_cols[:5]:  # Show first 5 for brevity
            types = X[col].dropna().apply(type).value_counts().to_dict()
            print(f"  - {col}: {types}")
        print("These will be converted to consistent types before processing.")

    # Define transformers with type handling and DataFrame preservation
    categorical_transformer = Pipeline(steps=[
        ('type_converter', DataTypeConverter(categorical_cols=categorical_features)),
        ('imputer', DataFrameImputer(strategy='constant', fill_value='missing')),
        ('onehot', DataFrameOneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    id_transformer = Pipeline(steps=[
        ('type_converter', DataTypeConverter(categorical_cols=id_features)),
        ('imputer', DataFrameImputer(strategy='constant', fill_value='0')),
        ('encoder', FrequencyEncoder(min_frequency=0.001))
    ])

    continuous_transformer = Pipeline(steps=[
        ('type_converter', DataTypeConverter(numeric_cols=continuous_features)),
        ('imputer', DataFrameImputer(strategy='median')),
        ('scaler', DataFrameRobustScaler())
    ])

    discrete_transformer = Pipeline(steps=[
        ('type_converter', DataTypeConverter(numeric_cols=discrete_features)),
        ('imputer', DataFrameImputer(strategy='median')),
        ('scaler', DataFrameStandardScaler())
    ])

    binary_transformer = Pipeline(steps=[
        ('type_converter', DataTypeConverter(categorical_cols=binary_features)),
        ('imputer', DataFrameImputer(strategy='constant', fill_value='missing')),
        ('encoder', DataFrameOneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown='ignore'))
    ])

    flag_transformer = Pipeline(steps=[
        ('type_converter', DataTypeConverter(numeric_cols=flag_features)),
        ('imputer', DataFrameImputer(strategy='constant', fill_value=0)),
        ('scaler', DataFrameStandardScaler())
    ])

    # Create the main preprocessing pipeline
    transformers = []

    # Only include feature groups with at least one column available
    available_features = [col for col in X.columns]

    if any(f in available_features for f in categorical_features):
        cat_cols = [f for f in categorical_features if f in X.columns]
        if cat_cols:
            transformers.append(('cat', categorical_transformer, cat_cols))

    if any(f in available_features for f in id_features):
        id_cols = [f for f in id_features if f in X.columns]
        if id_cols:
            transformers.append(('id', id_transformer, id_cols))

    if any(f in available_features for f in continuous_features):
        cont_cols = [f for f in continuous_features if f in X.columns]
        if cont_cols:
            transformers.append(('cont', continuous_transformer, cont_cols))

    if any(f in available_features for f in discrete_features):
        disc_cols = [f for f in discrete_features if f in X.columns]
        if disc_cols:
            transformers.append(('disc', discrete_transformer, disc_cols))

    if any(f in available_features for f in binary_features):
        bin_cols = [f for f in binary_features if f in X.columns]
        if bin_cols:
            transformers.append(('bin', binary_transformer, bin_cols))

    if any(f in available_features for f in flag_features):
        flag_cols = [f for f in flag_features if f in X.columns]
        if flag_cols:
            transformers.append(('flag', flag_transformer, flag_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop',
        verbose_feature_names_out=False
    )

    # Split data
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create a simpler initial pipeline
    print("\nTrying a simpler model first with DataFrame preservation...")
    simple_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=10,
            max_depth=5,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=1,
            class_weight='balanced'
        ))
    ])

    try:
        # Try fitting the simple pipeline first
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            simple_pipeline.fit(X_train, np.ravel(y_train))
        print("Simple model fit successfully!")

        X_test_transformed = simple_pipeline.named_steps['preprocessor'].transform(X_test)

        # If simple model works, proceed with hyperparameter tuning
        param_distributions = {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [5, 10, None],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 4],
            'classifier__class_weight': ['balanced', None]
        }

        # Configure RandomizedSearchCV
        search = RandomizedSearchCV(
            simple_pipeline,
            param_distributions=param_distributions,
            n_iter=5,
            cv=3,
            verbose=2,
            n_jobs=-1,
            random_state=42,
            scoring='f1_weighted',
            error_score=0
        )

        print("\nStarting model training and hyperparameter search...")
        search.fit(X_train, np.ravel(y_train))

        # Print results
        print("\nBest parameters:")
        print(search.best_params_)

        print("\nEvaluating on test set:")
        y_pred = search.predict(X_test)
        print(classification_report(y_test, y_pred))

        return search

    except Exception as e:
        print(f"\nError fitting model: {str(e)}")
        print("\nDiagnostic information:")

        # Try to identify where the transformation is failing
        for name, transformer, cols in preprocessor.transformers:
            print(f"\nChecking transformer: {name} for columns: {cols}")
            try:
                # Get the subset of data
                sub_X = X_train[cols]
                # Verify it's a DataFrame
                if not isinstance(sub_X, pd.DataFrame):
                    print(f"  WARNING: sub_X is not a DataFrame, it's {type(sub_X)}")
                    sub_X = pd.DataFrame(sub_X, columns=cols)

                print(f"  Data shape: {sub_X.shape}")
                print(f"  Data types: {sub_X.dtypes.to_dict()}")

                # Try transforming with each step individually
                current_X = sub_X
                for step_name, step in transformer.steps:
                    print(f"  Checking step: {step_name}")
                    try:
                        step.fit(current_X)
                        current_X = step.transform(current_X)
                        print(f"    Step {step_name} succeeded, output shape: {current_X.shape}")
                        print(f"    Output type: {type(current_X)}")
                        # If not a DataFrame, show data sample
                        if not isinstance(current_X, pd.DataFrame):
                            print(f"    Sample: {current_X[:2]}")
                    except Exception as step_e:
                        print(f"    Error in step {step_name}: {str(step_e)}")
                        # Try to provide more context
                        if isinstance(current_X, pd.DataFrame):
                            print(f"    Input DataFrame info: {current_X.info()}")
                        break

            except Exception as sub_e:
                print(f"  Error checking transformer {name}: {str(sub_e)}")

        # Return diagnostics
        return {"error": str(e)}




if __name__ == '__main__':
    cvBestModel = build_baseball_pipeline().best_estimator_

    # Get the fitted preprocessor
    fitted_preprocessor = cvBestModel.named_steps['preprocessor']

    # Get feature names after full transformation
    print("\n✅ Extracting transformed feature names...")
    feature_names = get_feature_names(fitted_preprocessor)

    # Extract feature importances from RandomForest
    rf_clf = cvBestModel.named_steps['classifier']
    importances = rf_clf.feature_importances_

    # Combine and sort
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    # Show top 20 features
    print("\n🔥 Top 20 Most Important Features:")
    print(feature_importance_df.head(20).to_string(index=False))