from BaseballDataVisualization1D import csv_to_df
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.feature_selection import f_classif, mutual_info_classif


def analyze_feature_relationships(df, target_col):
    """
    Automatically encodes features and computes f_classif and mutual_info_classif scores.

    Parameters:
        df (pd.DataFrame): The full DataFrame with features and target
        target_col (str): The name of the target column (must be categorical)

    Returns:
        pd.DataFrame: Sorted table with f_score, p_value, and mutual_info for each feature
    """
    # Separate target and features
    y_raw = df[target_col]
    X_raw = df.drop(columns=[target_col])

    # Encode target
    y = LabelEncoder().fit_transform(y_raw)

    # Separate input features
    cat_cols = X_raw.select_dtypes(include=['object', 'category']).columns
    num_cols = X_raw.select_dtypes(include=[np.number]).columns

    X_cat = X_raw[cat_cols]
    X_num = X_raw[num_cols]

    # Encode categorical features
    if not X_cat.empty:
        X_cat_encoded = OrdinalEncoder().fit_transform(X_cat)
    else:
        X_cat_encoded = np.empty((len(df), 0))

    # Combine encoded categorical and numeric features
    X_full = np.hstack([X_cat_encoded, X_num.to_numpy()])
    feature_names = list(cat_cols) + list(num_cols)

    # Mark which features are discrete
    discrete_flags = [True] * X_cat_encoded.shape[1] + [False] * X_num.shape[1]

    # Compute scores
    f_vals, p_vals = f_classif(X_full, y)
    mi_vals = mutual_info_classif(X_full, y, discrete_features=discrete_flags)

    # Build result DataFrame
    results_df = pd.DataFrame({
        'feature': feature_names,
        'f_score': f_vals,
        'f_p_value': p_vals,
        'mutual_info': mi_vals
    }).sort_values(by='mutual_info', ascending=False)

    return results_df




def display_correlation_matrix(data):
    non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
    print("Non-numeric columns excluded from correlation:", non_numeric_cols.tolist())

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    numeric_df = data.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    # Draw the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap='coolwarm', center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5})
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def drop_highly_correlated_features(data, threshold=0.95):
    # Select numeric features only
    numeric_df = data.select_dtypes(include=[np.number])

    # Compute the correlation matrix
    corr_matrix = numeric_df.corr().abs()

    # Upper triangle of the correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation above the threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Drop them from the original DataFrame
    reduced_df = data.drop(columns=to_drop)

    return reduced_df, to_drop




if __name__ == '__main__':
    df = csv_to_df()

    display_correlation_matrix(df)

    threshold = .95
    reducedDf, toDrop = drop_highly_correlated_features(data=df,threshold=threshold)

    print("numeric features with correlation >= threshold,",threshold,": ",toDrop)

    display_correlation_matrix(reducedDf)

    #print(analyze_feature_relationships(df, 'description'))
