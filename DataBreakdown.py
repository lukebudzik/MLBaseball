import BaseballDataVisualization1D as bbdata
import pandas as pd
import numpy as np


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
