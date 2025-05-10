import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime


# Import from existing modules
from BaseballDataVisualization1D import csv_to_df

# Import custom functions from our enhanced pipeline
# Assumes all the code artifacts have been integrated into Python modules

def cross_val_fit(pipeline):
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

    # Define parameter grid for Random Forest
    param_grid = {
        'clf__n_estimators': [100, 200, 300],
        'clf__max_depth': [None, 10, 20, 30],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4]
    }

    # Use RandomizedSearchCV for efficiency
    random_search = RandomizedSearchCV(
        pipeline, param_distributions=param_grid,
        n_iter=20, cv=5, scoring='f1_weighted',
        n_jobs=-1, random_state=42, verbose=1
    )
    random_search.fit(X_train, y_train)

    # Get best parameters and model

    return random_search.best_params_,random_search.best_estimator_


from sklearn.feature_selection import SelectFromModel, RFECV


# Method 1: Feature importance-based selection
def importance_based_selection(pipeline, X_train, y_train, X_test):
    # First fit the pipeline to get feature importances
    pipeline.fit(X_train, y_train)

    # Get feature importances from the Random Forest
    feature_importances = pipeline.named_steps['clf'].feature_importances_

    # Create a selector using the median importance as threshold
    selector = SelectFromModel(pipeline.named_steps['clf'], threshold='median')

    # Apply preprocessing from pipeline to get transformed features
    X_train_preprocessed = pipeline.named_steps['preprocessor'].transform(X_train)

    # Fit the selector
    selector.fit(X_train_preprocessed, y_train)

    # Transform the data
    X_train_selected = selector.transform(X_train_preprocessed)

    # Report feature reduction
    print(f"Selected {X_train_selected.shape[1]} features out of {X_train_preprocessed.shape[1]}")

    return selector


# Method 2: Recursive Feature Elimination with Cross-Validation
def rfe_selection(pipeline, X_train, y_train, cv=5):
    # Create a new pipeline with only preprocessing
    preprocess_pipeline = Pipeline(steps=[
        ('preprocessor', pipeline.named_steps['preprocessor'])
    ])

    # Apply preprocessing
    X_train_preprocessed = preprocess_pipeline.fit_transform(X_train)

    # Set up RFECV with the classifier
    rfecv = RFECV(
        estimator=pipeline.named_steps['clf'],
        step=1,
        cv=cv,
        scoring='f1_weighted',
        min_features_to_select=10,
        n_jobs=-1,
        verbose=1
    )

    # Fit RFECV
    rfecv.fit(X_train_preprocessed, y_train)

    # Report results
    print(f"Optimal number of features: {rfecv.n_features_}")

    return rfecv


from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_with_cross_validation(pipeline, X, y, cv=5, n_repeats=3):
    """
    Evaluate the model using repeated stratified k-fold cross-validation
    with multiple metrics
    """
    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'f1_weighted': 'f1_weighted',
        'precision_weighted': 'precision_weighted',
        'recall_weighted': 'recall_weighted',
        'roc_auc_ovr': 'roc_auc_ovr' if len(np.unique(y)) > 2 else 'roc_auc'
    }

    # Set up cross-validation strategy
    cv_strategy = RepeatedStratifiedKFold(n_splits=cv, n_repeats=n_repeats, random_state=42)

    # Perform cross-validation
    from sklearn.model_selection import cross_validate
    cv_results = cross_validate(
        pipeline, X, y,
        cv=cv_strategy,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )

    # Print results
    print("\n=== Cross-Validation Results ===")
    for metric in scoring.keys():
        train_scores = cv_results[f'train_{metric}']
        test_scores = cv_results[f'test_{metric}']

        print(f"{metric}:")
        print(f"  Training: {train_scores.mean():.4f} ± {train_scores.std():.4f}")
        print(f"  Testing:  {test_scores.mean():.4f} ± {test_scores.std():.4f}")

    return cv_results


from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import time
from sklearn.ensemble import RandomForestClassifier


def compare_models(X_train, X_test, y_train, y_test, preprocessor):
    """Compare different classifiers with the same preprocessing pipeline"""
    # Define models to test
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'ExtraTrees': ExtraTreesClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(random_state=42)
    }

    # Preprocess data once
    X_train_prep = preprocessor.transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    results = {}

    for name, model in models.items():
        start_time = time.time()
        print(f"\nTraining {name}...")

        # Train the model
        model.fit(X_train_prep, y_train)

        # Evaluate on test set
        y_pred = model.predict(X_test_prep)

        # Calculate metrics
        train_time = time.time() - start_time
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
            'training_time': train_time
        }

        print(f"{name} trained in {train_time:.2f} seconds")
        print(f"Accuracy: {results[name]['accuracy']:.4f}")
        print(f"F1 Score (weighted): {results[name]['f1_weighted']:.4f}")

    return results


import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class BaseballFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for baseball-specific feature engineering"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Make a copy to avoid modifying the original dataframe
        X_transformed = X.copy()

        # Feature 1: Calculate score difference (home - away)
        if 'home_score' in X_transformed.columns and 'away_score' in X_transformed.columns:
            X_transformed['score_diff'] = X_transformed['home_score'] - X_transformed['away_score']

        # Feature 2: Count baserunners (0-3)
        if all(col in X_transformed.columns for col in ['on_1b', 'on_2b', 'on_3b']):
            X_transformed['baserunners_count'] = (
                    X_transformed['on_1b'].notna().astype(int) +
                    X_transformed['on_2b'].notna().astype(int) +
                    X_transformed['on_3b'].notna().astype(int)
            )

        # Feature 3: Calculate count leverage (balls-strikes ratio)
        if 'balls' in X_transformed.columns and 'strikes' in X_transformed.columns:
            X_transformed['count_leverage'] = (X_transformed['balls'] + 1) / (X_transformed['strikes'] + 1)

        # Feature 4: Calculate pitch velocity to spin ratio
        if 'release_speed' in X_transformed.columns and 'release_spin_rate' in X_transformed.columns:
            X_transformed['speed_spin_ratio'] = X_transformed['release_spin_rate'] / (
                        X_transformed['release_speed'] + 0.1)

        # Feature 5: Calculate pitch break magnitude
        if 'pfx_x' in X_transformed.columns and 'pfx_z' in X_transformed.columns:
            X_transformed['break_magnitude'] = np.sqrt(X_transformed['pfx_x'] ** 2 + X_transformed['pfx_z'] ** 2)

        # Feature 6: Calculate pitch location from center of zone
        if all(col in X_transformed.columns for col in ['plate_x', 'plate_z', 'sz_top', 'sz_bot']):
            # Calculate center of strike zone
            X_transformed['sz_center'] = (X_transformed['sz_top'] + X_transformed['sz_bot']) / 2
            # Distance from center (x=0, z=sz_center)
            X_transformed['pitch_location_from_center'] = np.sqrt(
                X_transformed['plate_x'] ** 2 +
                (X_transformed['plate_z'] - X_transformed['sz_center']) ** 2
            )

        # Feature 7: Inning phase (early 1-3, middle 4-6, late 7+)
        if 'inning' in X_transformed.columns:
            X_transformed['inning_phase'] = pd.cut(
                X_transformed['inning'],
                bins=[0, 3, 6, np.inf],
                labels=['early', 'middle', 'late']
            )

        # Feature 8: Days rest impact (combination of days since prev game)
        if 'pitcher_days_since_prev_game' in X_transformed.columns:
            X_transformed['pitcher_rest_impact'] = pd.cut(
                X_transformed['pitcher_days_since_prev_game'],
                bins=[-1, 3, 5, np.inf],
                labels=['short_rest', 'normal_rest', 'extended_rest']
            )

        # Feature 9: Count state combinations
        if 'balls' in X_transformed.columns and 'strikes' in X_transformed.columns:
            X_transformed['count_state'] = X_transformed['balls'].astype(str) + '-' + X_transformed['strikes'].astype(
                str)

        # Feature 10: Age difference (pitcher - batter)
        if 'age_pit' in X_transformed.columns and 'age_bat' in X_transformed.columns:
            X_transformed['age_diff'] = X_transformed['age_pit'] - X_transformed['age_bat']

        # Return the transformed dataframe
        return X_transformed


def enhance_pipeline_with_feature_engineering(original_pipeline):
    """Add feature engineering step to the beginning of a pipeline"""

    # Extract the preprocessor from the original pipeline
    preprocessor = original_pipeline.named_steps['preprocessor']
    classifier = original_pipeline.named_steps['clf']

    # Create a new pipeline with feature engineering
    enhanced_pipeline = Pipeline(steps=[
        ('feature_engineering', BaseballFeatureEngineer()),
        ('preprocessor', preprocessor),
        ('clf', classifier)
    ])

    return enhanced_pipeline


from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter


def check_class_distribution(y):
    """Check distribution of classes in the target variable"""
    class_counts = Counter(y)
    print("Class distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} ({count / len(y) * 100:.2f}%)")

    # Calculate imbalance ratio
    majority_class_count = max(class_counts.values())
    minority_class_count = min(class_counts.values())
    imbalance_ratio = majority_class_count / minority_class_count
    print(f"Imbalance ratio: {imbalance_ratio:.2f}")

    return class_counts


def create_balanced_pipeline(preprocessor, classifier, balancing_strategy='smote', sampling_strategy='auto'):
    """Create a pipeline with balancing step"""

    if balancing_strategy == 'smote':
        balancer = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    elif balancing_strategy == 'oversample':
        balancer = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
    elif balancing_strategy == 'undersample':
        balancer = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    else:
        raise ValueError("Unknown balancing strategy. Use 'smote', 'oversample', or 'undersample'")

    # Create imbalanced-learn pipeline
    balanced_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('balancer', balancer),
        ('classifier', classifier)
    ])

    return balanced_pipeline


def compare_balancing_strategies(X_train, y_train, X_test, y_test, preprocessor, classifier):
    """Compare different balancing strategies"""

    # Check original class distribution
    print("Original class distribution:")
    check_class_distribution(y_train)

    # Define balancing strategies to test
    strategies = {
        'No balancing': None,
        'SMOTE': 'smote',
        'Random Oversampling': 'oversample',
        'Random Undersampling': 'undersample'
    }

    results = {}

    for name, strategy in strategies.items():
        print(f"\nEvaluating: {name}")

        if strategy is None:
            # Standard pipeline without balancing
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', classifier)
            ])
        else:
            # Create balanced pipeline
            pipeline = create_balanced_pipeline(preprocessor, classifier, strategy)

        # Train the pipeline
        pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = pipeline.predict(X_test)

        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted')
        }

        print(f"Accuracy: {results[name]['accuracy']:.4f}")
        print(f"F1 Score (weighted): {results[name]['f1_weighted']:.4f}")

    return results


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve,
    precision_recall_curve, auc, roc_auc_score
)


def evaluate_model_performance(model, X_test, y_test, class_names=None):
    """Comprehensive model evaluation"""

    # Get predictions
    y_pred = model.predict(X_test)

    # Get probability predictions if available
    try:
        y_prob = model.predict_proba(X_test)
    except:
        y_prob = None

    # 1. Classification Report
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

    # 3. ROC Curve and AUC (for multi-class, one-vs-rest)
    if y_prob is not None:
        n_classes = len(np.unique(y_test))

        if n_classes == 2:  # Binary classification
            # Calculate ROC curve and ROC area
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                     label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.show()

            # Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y_test, y_prob[:, 1])
            pr_auc = auc(recall, precision)

            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', lw=2,
                     label=f'Precision-Recall curve (area = {pr_auc:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="lower left")
            plt.show()

        else:  # Multi-class
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            # Convert y_test to one-hot encoding for ROC calculation
            y_test_bin = pd.get_dummies(y_test).values

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Plot ROC curves
            plt.figure(figsize=(10, 8))
            for i in range(n_classes):
                class_label = class_names[i] if class_names else i
                plt.plot(fpr[i], tpr[i], lw=2,
                         label=f'Class {class_label} (area = {roc_auc[i]:.2f})')

            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Multi-class ROC')
            plt.legend(loc="lower right")
            plt.show()

    # 4. Error Analysis - Find most frequently misclassified samples
    errors = X_test[y_test != y_pred].copy()
    errors['true_label'] = y_test[y_test != y_pred]
    errors['predicted_label'] = y_pred[y_test != y_pred]

    print("\n=== Error Analysis ===")
    error_counts = pd.crosstab(errors['true_label'], errors['predicted_label'],
                               rownames=['True'], colnames=['Predicted'])
    print("Most frequent misclassifications:")
    print(error_counts)

    return {
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': cm
    }


def feature_importance_analysis(model, feature_names):
    """Analyze feature importances from tree-based models"""

    # Check if model has feature_importances_ attribute (tree-based models)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Create a DataFrame for better visualization
        importance_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in indices],
            'Importance': importances[indices]
        })

        # Plot feature importances
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        plt.show()

        return importance_df
    else:
        print("Model does not have feature_importances_ attribute.")
        return None


if __name__ == "__main__":
    # 1. Load the data
    print("Loading baseball data...")
    data = csv_to_df()

    # 2. Define features and target
    target = ['description']

    # Define feature groups based on your original code
    categorical_objects = [
        'pitch_type', 'stand', 'zone',
        'p_throws', 'home_team', 'away_team', 'inning_topbot', 'pitch_name',
        'if_fielding_alignment', 'of_fielding_alignment'
    ]
    identifying_objects = [
        'game_date', 'fielder_2', 'fielder_3', 'fielder_4', 'fielder_5',
        'fielder_6', 'fielder_7', 'fielder_8', 'fielder_9', 'batter',
        'pitcher', 'game_pk'
    ]
    id_presence_cols = ['on_1b', 'on_2b', 'on_3b']
    numeric_discrete_features = [
        'balls', 'strikes', 'outs_when_up', 'inning', 'n_thruorder_pitcher',
        'n_priorpa_thisgame_player_at_bat', 'at_bat_number', 'pitch_number',
        'home_score', 'away_score', 'bat_score', 'fld_score', 'spin_axis',
        'age_pit', 'age_bat', 'pitcher_days_since_prev_game',
        'batter_days_since_prev_game', 'pitcher_days_until_next_game',
        'batter_days_until_next_game',
    ]
    numeric_continuous_features = [
        'release_speed', 'release_pos_x', 'release_pos_z', 'pfx_x', 'pfx_z',
        'plate_x', 'plate_z', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
        'sz_top', 'sz_bot', 'effective_speed', 'release_spin_rate',
        'release_extension', 'release_pos_y', 'api_break_z_with_gravity',
        'api_break_x_arm', 'api_break_x_batter_in', 'arm_angle'
    ]

    # Combine all feature groups
    nonexcluded = categorical_objects + identifying_objects + id_presence_cols + numeric_discrete_features + numeric_continuous_features

    # Ensure target is not in features
    groupedInputFeatures = [f for f in nonexcluded if f not in target]

    # 3. Split the data
    y = data[target[0]]
    X = data[groupedInputFeatures]

    print(f"Data shape: {X.shape} with target: {target[0]}")
    print(f"Target distribution:\n{y.value_counts(normalize=True)}")

    # Create stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Execute enhanced pipeline building
    print("Building enhanced machine learning pipeline...")
    results = build_enhanced_pipeline(X_train, y_train, X_test, y_test)

    # 5. Print summary of results
    print("\n" + "=" * 50)
    print(f"Model training complete! Files saved to: {results['output_dir']}")
    print("=" * 50)
    print(f"Best parameters: {results['best_params']}")
    print("\nModel Performance Comparison:")
    print(results['model_performance'])

    # 6. Feature importance plot (if available)
    importance_file = os.path.join(results['output_dir'], "feature_importances.png")
    if os.path.exists(importance_file):
        from IPython.display import Image

        print("\nTop Feature Importances:")
        display(Image(filename=importance_file))

    print("\nTo use the trained model for prediction:")
    print(f"model = joblib.load('{results['output_dir']}/baseball_prediction_model.joblib')")
    print("predictions = model.predict(new_data)")