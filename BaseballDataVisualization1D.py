import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

STATCAST_DATA = '2024regSeason.csv'

def csv_to_df(csv):
    df = pd.read_csv(csv)
    print("number of features in 2024 season dataset:", {df.shape[1]} )

    return df

def visualize_vars(df):
    num_cols = df.select_dtypes(include='number').columns
    cols_per_plot = 20  # Adjust to 3 or 2 if still crowded

    for i in range(0, len(num_cols), cols_per_plot):
        subset = num_cols[i:i + cols_per_plot]
        df[subset].hist(bins=30, figsize=(16, 8))  # Bigger figure
        plt.tight_layout()
        plt.show()

    print("number of numerical features shown with histograms:", {len(num_cols)} )


    threshold = 50  # You can adjust this for the number of categories a variable has for
                # high/low cardinality
    cat_cols = df.select_dtypes(include='object').columns
    print("number of categorical features:", len(cat_cols) )

    # Filter columns with fewer than `threshold` unique values
    low_card_cols = [col for col in cat_cols if df[col].nunique() < threshold]

    print("Low-cardinality categorical columns shown with box-plots:", low_card_cols)

    high_card_cols = [col for col in cat_cols if df[col].nunique() >= threshold]
    print("High-cardinality columns skipped:", high_card_cols)


    cols_per_plot = 4

    for i in range(0, len(low_card_cols), cols_per_plot):
        subset = low_card_cols[i:i + cols_per_plot]
        fig, axes = plt.subplots(1, len(subset), figsize=(5 * len(subset), 5))

        if len(subset) == 1:
            axes = [axes]

        for col, ax in zip(subset, axes):
            df[col].value_counts().plot(kind='bar', ax=ax)
            ax.set_title(f'Count of {col}')
            ax.set_ylabel('Frequency')
            ax.set_xlabel(col)
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    visualize_vars(csv_to_df(STATCAST_DATA))




##  matplotlib displays all numerical features and their distributions as well
# as box plots for the low-cardinality categorical variables: those with less than 50 categories
# (the high-cardinality variables are dates of games in which these pitch-events take place, descriptions of the outcome of the play
# that the individual pitch was a part of, and player names)

##all the following is the output of this simple data visualization:
# number of features in dataset: {113}

# number of numerical features shown with histograms: {96}

# number of categorical features: {17}

# Low-cardinality categorical columns shown with box-plots:
# ['pitch_type', 'events', 'description', 'game_type', 'stand', 'p_throws', 'home_team',
# 'away_team', 'type', 'bb_type', 'inning_topbot', 'pitch_name', 'if_fielding_alignment',
# 'of_fielding_alignment']

# High-cardinality columns skipped: ['game_date', 'player_name', 'des']
##

##FOLLOWING THIS IS IMPLEMENTATION OF FEATURE vs TARGET VARIABLES IN: BaseballVisualization2.py