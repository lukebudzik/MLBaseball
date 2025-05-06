import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import chi2_contingency
from BaseballDataVisualization1D import csv_to_df
import warnings

warnings.filterwarnings('ignore')


def load_and_explore_statcast_data(missing_threshold=0.1):
    """Load and perform initial exploration of Statcast data."""
    print("Loading data...")
    df = csv_to_df()

    # Drop columns with excessive missing values
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    columns_to_drop = missing_percent[missing_percent > missing_threshold * 100].index.tolist()

    print(f"\nDropping {len(columns_to_drop)} columns with >{missing_threshold * 100}% missing values.")
    print(columns_to_drop)
    df = df.drop(columns=columns_to_drop)

    # Display basic info and target distribution
    print(f"\nDataset shape: {df.shape}")
    print(f"\nTarget variable 'description' distribution:")
    target_counts = df['description'].value_counts()
    print(target_counts)

    # Plot target distribution (only if it has fewer unique values)
    if len(target_counts) <= 20:
        plt.figure(figsize=(12, 6))
        sns.countplot(y='description', data=df, order=target_counts.index)
        plt.title('Distribution of Target Variable: description')
        plt.tight_layout()
        plt.show()

    return df


def identify_id_columns(df, nunique=50):
    """Identify columns that are likely ID fields or have too many unique values."""
    id_like_columns = []
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'int64':
            unique = df[col].nunique()


            unique_ratio = unique / len(df)
            print(col,' ', unique)
            if (unique > nunique) and col != 'at_bat_number':
                print(col)
                id_like_columns.append(col)

    print(f"\nIdentified {len(id_like_columns)} potential ID columns.")
    return id_like_columns


def analyze_feature_relationships(df, exclude_columns=None):
    """Analyze feature relationships with the target variable."""
    if exclude_columns is None:
        exclude_columns = []

    analysis_df = df.copy()
    all_columns = set(analysis_df.columns) - set(exclude_columns) - {'description'}
    categorical_features = [col for col in all_columns if analysis_df[col].dtype == 'object']
    numerical_features = [col for col in all_columns if
                          col not in categorical_features and analysis_df[col].dtype in ['int64', 'float64']]

    print(f"\nCategorical features: {len(categorical_features)}")
    print(f"Numerical features: {len(numerical_features)}")

    # Analyze mutual information for numerical features
    if numerical_features:
        print("\nCalculating mutual information scores for numerical features...")
        X_numeric = analysis_df[numerical_features]

        # Fill missing values only for numerical columns using the median
        X_numeric = X_numeric.fillna(X_numeric.median())

        mi_scores = mutual_info_classif(X_numeric, analysis_df['description'], random_state=42)
        mi_df = pd.DataFrame({'Feature': numerical_features, 'MI Score': mi_scores})
        mi_df = mi_df.sort_values('MI Score', ascending=False)

        print("\nTop 10 numerical features by mutual information:")
        print(mi_df.head(10))

    # Chi-square test for categorical features
    chi_results = []
    for feature in categorical_features:
        if analysis_df[feature].nunique() > 50:  # Skip features with too many categories
            continue

        contingency_table = pd.crosstab(analysis_df[feature], analysis_df['description'])
        if contingency_table.size == 0 or np.any(contingency_table.values < 5):
            continue

        chi2, p, dof, expected = chi2_contingency(contingency_table)
        chi_results.append((feature, chi2, p))

    chi_df = pd.DataFrame(chi_results, columns=['Feature', 'Chi-Square', 'P-Value'])
    chi_df = chi_df.sort_values('Chi-Square', ascending=False)

    print("\nTop categorical features by chi-square:")
    print(chi_df.head(10))

    return mi_df, chi_df


def identify_correlated_features(df, numerical_features, threshold=0.8):
    """Identify highly correlated features and visualize."""
    corr_df = df[numerical_features].fillna(df.median())
    corr_matrix = corr_df.corr().abs()

    # Extract upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_features = [column for column in upper.columns if any(upper[column] > threshold)]

    print(f"\nFound {len(high_corr_features)} features with correlation > {threshold}")

    # Plot correlation matrix for top 10 features
    top_var_features = corr_df.var().sort_values(ascending=False).head(10).index
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df[top_var_features].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of Top 10 Features by Variance')
    plt.tight_layout()
    plt.show()

    return high_corr_features


def handle_player_ids(df, id_columns):
    """Handle player ID columns (either drop or transform)."""
    if id_columns:
        print("\nHandling player ID columns...")
        for col in id_columns:
            print(f"Consider dropping ID-like column: {col}")
            df.drop(columns=[col], inplace=True)
        return df
    else:
        print("No ID columns identified.")


def summarize_recommendations(mi_df, chi_df, high_corr_features, id_columns):
    """Summarize recommendations for feature selection."""
    print("\n=== Feature Selection Recommendations ===")

    # Top features by importance
    if not mi_df.empty:
        print("Top numerical features by mutual information:")
        for i, (feature, score) in enumerate(zip(mi_df['Feature'], mi_df['MI Score'])):
            print(f"{i + 1}. {feature}: {score:.4f}")

    if not chi_df.empty:
        print("\nTop categorical features by chi-square:")
        for i, (feature, score) in enumerate(zip(chi_df['Feature'], chi_df['Chi-Square'])):
            print(f"{i + 1}. {feature}: {score:.2f}")

    # Features to drop
    if id_columns:
        print(f"\nID-like columns to drop:")
        for col in id_columns:
            print(f"- {col}")

    if high_corr_features:
        print(f"\nHighly correlated features to consider dropping:")
        for col in high_corr_features:
            print(f"- {col}")


# Main execution
if __name__ == "__main__":
    # Load and explore data - now drops columns with >30% missing values
    df = load_and_explore_statcast_data(missing_threshold=0.1)

    # Identify ID-like columns
    id_columns = identify_id_columns(df, nunique=50)

    # Handle player IDs
    df = handle_player_ids(df, id_columns)

    # Analyze feature relationships with target, excluding ID columns
    mi_scores, chi_scores = analyze_feature_relationships(df, exclude_columns=id_columns)

    # Identify correlated features
    numerical_features = [col for col in df.select_dtypes(include=['int64', 'float64']).columns if col not in id_columns]
    high_corr_features = identify_correlated_features(df, numerical_features, threshold=0.8)

    # Summarize all recommendations
    summarize_recommendations(mi_scores, chi_scores, high_corr_features, id_columns)
