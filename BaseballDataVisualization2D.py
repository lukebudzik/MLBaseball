import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import BaseballDataVisualization1D as bb
from BaseballDataVisualization1D import STATCAST_DATA


def plot_target_vs_features(df, target, features):
    for feature in features:
        plt.figure(figsize=(10, 6))
        print(f'plotting ', feature)
        if df[feature].dtype in ['int64', 'float64']:  # Numerical feature
            sns.boxplot(data=df, x=target, y=feature)
            plt.title(f'{feature} distribution by {target}')
            plt.xticks(rotation=45, ha='right')
        else:  # Categorical feature
            # Use normalized stacked bar plot
            cross_tab = pd.crosstab(df[feature], df[target], normalize='index')
            cross_tab.plot(kind='bar', stacked=True, figsize=(10, 6))
            plt.title(f'{feature} vs {target} (proportion)')
            plt.ylabel('Proportion')
            plt.legend(title=target, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
        plt.grid(False)
        plt.show()


if __name__ == '__main__':
    data = bb.csv_to_df()
    columns = ['description','game_date','player_name', 'events','game_year','bb_type','spin_dir',
               'spin_rate_deprecated', 'break_angle_deprecated','des','game_type','hit_location',
               'break_length_deprecated', 'tfs_deprecated', 'tfs_zulu_deprecated', 'umpire',
               'sv_id','hc_x','hc_y','hit_distance_sc','launch_speed','launch_angle','game_pk',
               'estimated_ba_using_speedangle','estimated_woba_using_speedangle','woba_value',
               'woba_denom','babip_value','iso_value','launch_speed_angle','post_away_score',
               'post_home_score','post_bat_score','post_fld_score','delta_home_win_exp',
               'delta_run_exp','estimated_slg_using_speedangle','delta_pitcher_run_exp',
               'hyper_speed','home_score_diff','bat_score_diff','home_win_exp',
               'bat_win_exp','age_pit_legacy','age_bat_legacy','n_thruorder_pitcher','type']
    inputData = data.drop(columns=columns)
    print(f'input data columns:', inputData.columns)
    #plot_target_vs_features(data, 'description', inputData)




    # Heatmap
    plt.figure(figsize=(14, 6))
    sns.heatmap(inputData.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.xlabel('Columns')
    plt.tight_layout()
    plt.show()

    missing_counts = inputData.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    missing_counts.plot(kind='bar')
    plt.title('Number of Missing Values per Column')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

