# data science
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import normaltest

# algorithms
import math
import itertools

# machine learning
from sklearn.base import is_classifier, clone
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import root_mean_squared_error, r2_score, log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from xgboost import XGBRegressor, XGBClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from bayes_opt import BayesianOptimization

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# display
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
sns.set(style='whitegrid', font='Average')

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# global vars
ROOT = './data/'
MENS_ROOT = './data/mens/'
WOMENS_ROOT = './data/womens/'

# set numpy seed
SEED = 9
np.random.seed(SEED)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# eda_compact.ipynb

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def show_shape_and_nulls(df):
    """
    Display the shape of a DataFrame and the number of null values in each column.

    Args:
    - df (pd.DataFrame): The DataFrame to analyze.

    Returns:
    - None
    """

    # print shape
    print(f'Shape: {df.shape}')

    # check for missing values
    print('Null values:')

    # display null values
    display(df.isnull().sum().to_frame().T)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def show_unique_vals(df):
    """
    Print the number of unique values for each column in a DataFrame.
    If a column has fewer than 20 unique values, print those values.

    Args:
    - df (pd.DataFrame): The DataFrame to analyze.

    Returns:
    - None
    """

    # iterate over columns
    for col in df.columns:
        # get number of unique values and print
        n = df[col].nunique()
        print(f'"{col}" has {n} unique values')

        # if number of unique values is under 20, print the unique values
        if n < 20:
            print(df[col].unique())
        print()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def reshape_data(df):
    """
    Reshape the data to have each game counted for both teams.

    Args:
    - df (pd.DataFrame): DataFrame containing the data.

    Returns:
    - pd.DataFrame: Reshaped DataFrame.
    """

    # add 'LLoc' col
    df['LLoc'] = df['WLoc'].apply(lambda x: 'H' if x == 'A' else 'A' if x == 'H' else 'N')

    # rename maps for winners and losers
    win_map = {col: col[1:] if col.startswith('W') else (col[1:] + '_opp' if col.startswith('L') else col) for col in df.columns}
    lose_map = {col: col[1:] if col.startswith('L') else (col[1:] + '_opp' if col.startswith('W') else col) for col in df.columns}

    # reshape data to have each game counted for both teams
    df = pd.concat([df.rename(columns=win_map).assign(Win=1), df.rename(columns=lose_map).assign(Win=0)])

    return df

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def split_genders(df, id_col):
    """
    Split the df into mens and womens data.

    Args:
    - df (pd.DataFrame): DataFrame containing the data.
    - id_col (str): Column name containing the team ID.

    Returns:
    - df_mens (pd.DataFrame): DataFrame containing mens data.
    - df_womens (pd.DataFrame): DataFrame containing womens data
    """

    # error handling
    if id_col not in df.columns:
        raise ValueError(f"Column '{id_col}' not found in DataFrame. Available columns: {df.columns}")

    # split mens and womens data
    df_mens = df[df[id_col] < 3000].reset_index(drop=True)
    df_womens = df[df[id_col] > 3000].reset_index(drop=True)

    return df_mens, df_womens

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def plot_overlapping_histograms(df1, df2, columns=None, labels=('Men\'s', 'Women\'s'), colors=None):
    """
    Plot overlapping histograms for two DataFrames.

    Args:
    - df1 (pd.DataFrame): First DataFrame.
    - df2 (pd.DataFrame): Second DataFrame.
    - columns (list, optional): Columns to plot. Default is None.
    - labels (tuple, optional): Labels for the two DataFrames. Default is ('Men\'s', 'Women\'s').
    - colors (list, optional): Colors for the histograms. Default is None.

    Returns:
    - None
    """

    # check for missing columns
    if columns is None:
        columns = list(set(df1.columns) & set(df2.columns))  

    # check for missing colors
    if colors is None:
        colors = ['blue', 'red']

    # plot histograms
    num_cols = len(columns)
    rows = math.ceil(num_cols / 2)  

    # create subplots
    fig, axes = plt.subplots(rows, 2, figsize=(12, 4 * rows))
    axes = axes.flatten() if num_cols > 1 else [axes]

    # iterate over columns
    for i, col in enumerate(columns):
        # get min and max values to set bins
        min_value = min(df1[col].min(), df2[col].min())
        max_value = max(df1[col].max(), df2[col].max())
        bins = np.linspace(min_value, max_value, 30)

        # plot histograms
        sns.histplot(df1[col], kde=True, color=colors[0], label=labels[0], alpha=0.5, bins=bins, stat='percent', common_norm=True, ax=axes[i])
        sns.histplot(df2[col], kde=True, color=colors[1], label=labels[1], alpha=0.5, bins=bins, stat='percent', common_norm=True, ax=axes[i])
        
        # set title and legend
        axes[i].set_title(col)
        axes[i].legend()

    # remove extra axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def aggregate_compact_stats(df):
    """
    Aggregate team stats for each season on the compact dataset.

    Args:
    - df (pd.DataFrame): DataFrame containing the data.

    Returns:
    - team_stats (pd.DataFrame): DataFrame containing aggregated team stats.
    """

    # convert to polars
    df_pl = pl.from_pandas(df)

    # Compute team stats efficiently
    team_stats_pl = (df_pl.group_by(["Season", "TeamID"]).agg([
            # wins
            pl.len().alias("num_games"),
            pl.col("Win").mean().alias("win_pct"),

            # points
            pl.col("Score").mean().alias("mean_pts"),
            pl.col("Score").std().alias("std_pts"),
            pl.col("Score_opp").mean().alias("mean_pts_against"),
            pl.col("Score_opp").std().alias("std_pts_against"),
            pl.col("ScoreDiff").mean().alias("mean_score_diff"),
            pl.col("ScoreDiff").std().alias("std_score_diff"),

            # location
            (pl.col("Loc") == "H").mean().alias("home_game_pct"),
            (pl.col("Loc") == "A").mean().alias("away_game_pct"),
            (pl.col("Loc") == "N").mean().alias("neutral_game_pct"),
            pl.col("Win").filter(pl.col("Loc") == "H").mean().alias("home_win_pct"),
            pl.col("Win").filter(pl.col("Loc") == "A").mean().alias("away_win_pct"),
            pl.col("Win").filter(pl.col("Loc") == "N").mean().alias("neutral_win_pct"),

            # # one-score game stats
            # (pl.col("ScoreDiff").abs() <= 3).mean().alias("one_score_pct"),
            # pl.col("Win").filter(pl.col("ScoreDiff").abs() <= 3).mean().alias("one_score_win_pct"),

            # # overtime stats
            # (pl.col("NumOT") > 0).mean().alias("ot_pct"),
            # pl.col("Win").filter(pl.col("NumOT") > 0).mean().alias("ot_win_pct")

            # one-score + OT games
            ((pl.col("ScoreDiff").abs() <= 3) | (pl.col("NumOT") > 0)).mean().alias("close_games_pct"),
            pl.col("Win").filter((pl.col("ScoreDiff").abs() <= 3) | (pl.col("NumOT") > 0)).mean().alias("close_games_win_pct")]))

    # convert back to pandas
    return team_stats_pl.to_pandas()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def split_seed(seed):
    """
    Split the Seed value into Region, Seed number, and Play-in flag.

    Args:
    - seed (str): Seed value.

    Returns:
    - (tuple): Region, Seed number, Play-in flag.
    """

    return seed[0], seed[1:], (1 if len(seed[1:]) > 2 else 0)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def assign_tournament_round(df):
    """
    Creates a new column 'round' in the DataFrame that assigns a round number to each game. Assumes that play-in games are already filtered out.

    Args:
    - df (pd.DataFrame): DataFrame containing the data.

    Returns:
    - (pd.DataFrame): DataFrame with the 'round' column added
    """

    # assign round to 0 for all rows
    df['round'] = 0

    # split genders
    df_mens, df_womens = split_genders(df, id_col='WTeamID')

    # define round indices for standard seasons
    round_indices = [32, 48, 56, 60, 62, 63]

    # adjust for 2021 men's tournament (VCU-Oregon cancellation)
    special_case = {2021: [31, 47, 55, 59, 61, 62]}

    def assign_rounds(sub_df, special_case={}):
    # assign rounds
        for season, group in sub_df.groupby('Season'):
            # order by DayNum
            group = group.sort_values(by='DayNum')

            # get round indices
            indices = special_case.get(season, round_indices)

            # assign round numbers
            round_numbers = []
            prev_index = 0
            for r, i in enumerate(indices, 1):
                round_numbers.extend([r] * (i - prev_index))
                prev_index = i

            # ensure correct length
            if len(round_numbers) != len(group):
                raise ValueError(f"Mismatch in assigned rounds for season {season}")

            sub_df.loc[group.index, 'round'] = round_numbers
    
    # assign for men's and women's
    assign_rounds(df_mens, special_case)
    assign_rounds(df_womens)

    # return merged dfs
    return pd.concat([df_mens, df_womens], ignore_index=True)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def plot_upset_percentage(upsets, col):
    """
    Plot the year-over-year upset percentage and upset sum.

    Args:
    - upsets (pd.DataFrame): DataFrame containing upset percentage and sum.
    - col (str): Column to plot.

    Returns:
    - None
    """

    # create figure and first axis
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # plot regular upset percentage
    blue = '#215bfc'
    ax1.plot(upsets[col], upsets['Upset_Percentage'], marker='o', label='Upset Percentage', color=blue)
    ax1.set_xlabel(col)
    ax1.set_ylabel('Upset Percentage', color=blue)
    ax1.tick_params(axis='y', labelcolor=blue)

    # create second axis for upset sum
    ax2 = ax1.twinx()
    orange = '#fc5721'
    ax2.plot(upsets[col], upsets['Upset_Sum'], marker='s', linestyle='--', label='Sum of Seed Diffs', color=orange)
    ax2.set_ylabel('Sum of Seed Diffs', color=orange)
    ax2.tick_params(axis='y', labelcolor=orange)

    # combine legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# preds_compact.ipynb

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def get_dummy_preds(data):
    """
    Generate dummy predictions based on seed and win ratio.

    Args:
    - data (pd.DataFrame): DataFrame containing the data.

    Returns:
    - (np.array): An array of dummy predictions.
    """

    # create a container
    dummy_preds = []

    # iterate over rows
    for idx, row in data.iterrows():

        # if seed is lower, predict 1
        if data.loc[idx, "Seed_num_x"] < data.loc[idx, "Seed_num_y"]:
            dummy_preds.append(1)
        elif data.loc[idx, "Seed_num_x"] > data.loc[idx, "Seed_num_y"]:
            dummy_preds.append(0)

        # if seeds are equal, use win ratio
        else:
            if data.loc[idx, "win_pct_x"] > data.loc[idx, "win_pct_y"]:
                dummy_preds.append(1)
            else:
                dummy_preds.append(0)
    
    return np.array(dummy_preds)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def cross_val_model(estimator, df, target_col, gender, scaler, models_df, folds=10):
    """
    Perform KFold cross validation on a given estimator and store evaluation metrics.
    
    Args:
    - estimator (sklearn estimator): Estimator to use for modeling.
    - df (pd.DataFrame): Data to model.
    - target_col (str): Name of the target column ('win_x' for classification, 'score_diff_adj_x' for regression).
    - gender (str): 'm' or 'w'.
    - scaler (sklearn scaler, optional): Scaler to use for data. Default is None.
    - models_df (pd.DataFrame): DataFrame to save model results to. Expected columns: ['Gender', 'Model', 'Model_Params', 'Scaler', 'Num_Features, 'Train_R2', 'Val_R2', 'Train_RMSE', 'Val_RMSE', 'Train_LogLoss', 'Val_LogLoss', 'Train_Acc', 'Val_Acc']
    - folds (int): Number of cross-validation folds to use. Default is 10.
    
    Returns:
    - models_df (pd.DataFrame): Updated DataFrame with a new row containing model evaluation metrics.
    """
    
    # check if the estimator is a classifier
    if is_classifier(estimator):
        target_type = 'classification'
        X = df.drop(columns=['score_diff_adj_x', 'win_x'])
        y = df[target_col]

    else:
        # regression
        target_type = 'regression'
        X = df.drop(columns=['score_diff_adj_x', 'win_x'])
        y = df['score_diff_adj_x']
        y_accuracy = df['winx']
    
    # init lists to store metrics across folds
    train_r2_list, val_r2_list = [], []
    train_rmse_list, val_rmse_list = [], []
    train_logloss_list, val_logloss_list = [], []
    train_acc_list, val_acc_list = [], []
    
    kf = KFold(n_splits=folds, shuffle=True, random_state=SEED)
    
    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        
        if target_type == 'classification':
            # for classification, get the win_x column
            y_train = y.iloc[train_index]
            y_val = y.iloc[val_index]
        else:
            # for regression, get the score_diff_adj_x column
            y_train = y.iloc[train_index]
            y_val = y.iloc[val_index]

            # for accuracy in regression, get the win_x column.
            y_train_acc = y_accuracy.iloc[train_index]
            y_val_acc = y_accuracy.iloc[val_index]
        
        # scale data if a scaler is provided
        if scaler is not None:
            sc = clone(scaler)
            X_train = sc.fit_transform(X_train)
            X_val = sc.transform(X_val)
        else:
            # ensure we work with numpy arrays
            X_train = X_train.values
            X_val = X_val.values
        
        # train the model
        model = clone(estimator)
        model.fit(X_train, y_train)
        
        # predict on train and validation sets
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        if target_type == 'classification':
            # log loss
            train_loss = log_loss(y_train, y_train_pred)
            val_loss = log_loss(y_val, y_val_pred)

            # calculate accuracy
            train_acc = accuracy_score(y_train, y_train_pred)
            val_acc = accuracy_score(y_val, y_val_pred)

            # set other metrics to 0
            train_r2, val_r2, train_rmse, val_rmse = 0, 0, 0, 0
        else:
            # R-squared
            train_r2 = r2_score(y_train, y_train_pred)
            val_r2 = r2_score(y_val, y_val_pred)

            # RMSE
            train_rmse = root_mean_squared_error(y_train, y_train_pred)
            val_rmse = root_mean_squared_error(y_val, y_val_pred)

            # regression - convert continuous predictions to binary outcomes
            train_pred_win = (y_train_pred > 0).astype(int)
            val_pred_win = (y_val_pred > 0).astype(int)

            # calculate accuracy
            train_acc = accuracy_score(y_train_acc, train_pred_win)
            val_acc = accuracy_score(y_val_acc, val_pred_win)

            # set log loss to 0
            train_loss, val_loss = 0, 0
        
        # append metrics for this fold
        train_r2_list.append(train_r2)
        val_r2_list.append(val_r2)
        train_rmse_list.append(train_rmse)
        val_rmse_list.append(val_rmse)
        train_logloss_list.append(train_loss)
        val_logloss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
    
    # prepare a new row with averages
    new_row = {'Gender': gender,
        'Target': target_col,
        'Model': estimator.__class__.__name__,
        'Model_Params': str(estimator.get_params()),
        'Scaler': scaler.__class__.__name__,
        'Num_Features': X.shape[1],
        'Train_R2': np.mean(train_r2_list),
        'Val_R2': np.mean(val_r2_list),
        'Train_RMSE': np.mean(train_rmse_list),
        'Val_RMSE': np.mean(val_rmse_list),
        'Train_LogLoss': np.mean(train_logloss_list),
        'Val_LogLoss': np.mean(val_logloss_list),
        'Train_Acc': np.mean(train_acc_list),
        'Val_Acc': np.mean(val_acc_list)}

    # append the new row
    models_df.loc[len(models_df)] = new_row

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def evaluate_model(features_df, target_col, model, scaler, test_size=0.2):
    """
    Performs train-test split, fits the model, makes predictions, and returns predictions for evaluation.

    Args:
    - features_df (pd.DataFrame): dataframe with all features, including labels ('win_x' and 'score_diff_adj_x').
    - target_col (str): the target column to predict.
    - model (sklearn estimator): the model to fit and predict.
    - scaler (sklearn scaler): the scaler to use for feature scaling.
    - test_size (float, optional): proportion of data to use as test set. default is 0.2.

    Returns:
    - results_df (pd.DataFrame): dataframe with test features, predictions, and correctness.
    """

    # define X and y
    X = features_df.drop(columns=['score_diff_adj_x', 'win_x'])
    y = features_df[target_col]

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=SEED)

    # scale training and test data
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # fit the model on training data
    model.fit(X_train_scaled, y_train)

    # for regression models, convert predictions to binary outcomes
    if not is_classifier(model):
        # get predictions on the test set
        test_preds = model.predict(X_test_scaled)

        # convert predicted score_diff to binary for win/loss
        predicted_win = (test_preds > 0).astype(int)
        
        # get the actual win/loss values
        actual_win = features_df.loc[y_test.index]['win_x']
        
        # check if predicted win/loss matches the actual
        correct_mask = predicted_win == actual_win
    else:
        # for classification models, get the predicted probability for class 1
        test_preds = model.predict_proba(X_test_scaled)[:, 1]
        
        # round predicted probabilities to 0/1 and check correctness
        correct_mask = (test_preds.round() == y_test)

    # create a df to store test features and prediction results
    results_df = X_test.copy()
    results_df['Actual'] = y_test
    results_df['Prediction'] = test_preds

    # apply correct mask
    results_df['Correct'] = correct_mask

    # print accuracy
    print(f"Test Set ({test_size*100}%) Accuracy: {correct_mask.mean()*100:.2f}%")

    return results_df

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def get_seeding_diff_groups(preds):
    """
    Create 3 unique seed difference groups and print accuracy.

    Args:
    - preds (pd.DataFrame): DataFrame containing predictions.

    Returns:
    - upset_wins (pd.DataFrame): DataFrame containing upsets.
    - fav_wins (pd.DataFrame): DataFrame containing favorites.
    - equal_seeds (pd.DataFrame): DataFrame containing equal seeds.
    """

    # create seed_diff
    preds['seed_diff'] = np.abs(preds['Seed_num_x'] - preds['Seed_num_y'])

    # break into 3 different seed diff groups
    preds['seed_diff_group'] = pd.cut(preds['seed_diff'], bins=[0, 3, 6, 16], labels=['0-3', '4-6', '7+']).astype('category')

    # split into upsets, favorite, and equal seeds
    upset_wins = preds.query("((Seed_num_x > Seed_num_y) & (Actual > 0)) | ((Seed_num_x < Seed_num_y) & (Actual <= 0))")
    fav_wins = preds.query("((Seed_num_x > Seed_num_y) & (Actual <= 0)) | ((Seed_num_x < Seed_num_y) & (Actual > 0))")
    equal_seeds = preds.query("Seed_num_x == Seed_num_y")

    # iterate through upsets and favorites
    for i, data in enumerate([upset_wins, fav_wins]):
        if i == 0:
            print("Upsets:")
        else:
            print("Favorites:")

        # iterate through the 3 groups
        for group in ['0-3', '4-6', '7+']:
            # get the count of games in the group
            count = data.query("seed_diff_group == @group").shape[0]

            # get the accuracy of the group
            acc = data.query("seed_diff_group == @group")['Correct'].mean()

            # print
            print(f"Accuracy in games where seed diff is {group} ({count} games): {acc*100:.2f}%")
        print()

    # equal seeds
    equal_count = equal_seeds.shape[0]
    equal_acc = equal_seeds.query("Correct == 1").shape[0] / equal_count
    print(f"Accuracy in games where seeds are equal ({equal_count} games): {equal_acc*100:.2f}%")

    return upset_wins, fav_wins, equal_seeds

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def generate_matchups(seeds):
    """
    Generate all possible matchups between teams.

    Args:
    - seeds (pd.DataFrame): DataFrame containing seeds.

    Returns:
    - matchups (pd.DataFrame): DataFrame containing all possible matchups.
    """

    # split genders
    seeds_men, seeds_women = split_genders(seeds, id_col='TeamID')

    # create every possible matchup combo between seeds
    matchups_men, matchups_women = pd.DataFrame(), pd.DataFrame()
    matchups_men[['T1','T2']] = list(itertools.product(seeds_men['TeamID'], seeds_men['TeamID']))
    matchups_women[['T1','T2']] = list(itertools.product(seeds_women['TeamID'], seeds_women['TeamID']))

    # drop rows where T1 == T2
    matchups_men = matchups_men.query("T1 != T2").copy()
    matchups_women = matchups_women.query("T1 != T2").copy()

    # ensure T1 is always lower than T2
    matchups_men[['T1', 'T2']] = pd.DataFrame(np.sort(matchups_men[['T1', 'T2']].values, axis=1), index=matchups_men.index, columns=['T1', 'T2'])
    matchups_women[['T1', 'T2']] = pd.DataFrame(np.sort(matchups_women[['T1', 'T2']].values, axis=1), index=matchups_women.index, columns=['T1', 'T2'])

    # remove duplicate rows
    matchups_men = matchups_men.drop_duplicates()
    matchups_women = matchups_women.drop_duplicates()

    return matchups_men.sort_values(by=['T1', 'T2']), matchups_women.sort_values(by=['T1', 'T2'])

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def get_2025_predictions(features, matchups, model, scaler):
    """
    Generate predictions for the 2025 tournament.

    Args:
    - features (pd.DataFrame): DataFrame containing features.
    - matchups (pd.DataFrame): DataFrame containing matchups.
    - model (sklearn estimator): Model to use for predictions.
    - scaler (sklearn scaler): Scaler to use for features.

    Returns:
    - preds_x_df (pd.DataFrame): DataFrame containing predictions.
    """
    
    # merge features with matchups
    X = pd.merge(matchups, features, left_on='T1', right_on='TeamID', how='inner', suffixes=('_x', '_y'))
    X = pd.merge(X, features, left_on='T2', right_on='TeamID', how='inner').drop(columns=['T1', 'T2'])

    # sort cols
    X = X.reindex(sorted(X.columns), axis=1)

    # predict all matchups (that team x wins)
    X_scaled = scaler.transform(X.drop(columns=['TeamID_x', 'TeamID_y']))
    preds_x = model.predict_proba(X_scaled)[:, 0]

    # add team id cols back to preds
    preds_x_df = pd.DataFrame(preds_x, columns=['Pred'], index=X.index)
    preds_x_df = pd.concat([X[['TeamID_x', 'TeamID_y']], preds_x_df], axis=1)

    # create ID col
    preds_x_df['ID'] = '2025' + '_' + preds_x_df['TeamID_x'].astype(str) + '_' + preds_x_df['TeamID_y'].astype(str)

    # drop team id cols
    preds_x_df = preds_x_df[['ID', 'Pred']]

    return preds_x_df

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# eda_detailed.ipynb

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def aggregate_detailed_stats(df):
    """
    Aggregate team stats for each season on the detailed dataset.

    Args:
    - df (pd.DataFrame): DataFrame containing the data.

    Returns:
    - team_stats (pd.DataFrame): DataFrame containing aggregated team stats.
    """

    # convert to polars
    df_pl = pl.from_pandas(df)

    # compute team stats
    team_stats_pl = (df_pl.group_by(["Season", "TeamID"]).agg([
            # wins
            pl.len().alias("num_games"),
            pl.col("Win").mean().alias("win_pct"),

            # points
            pl.col("Score").mean().alias("mean_pts"),
            pl.col("Score").std().alias("std_pts"),
            pl.col("Score_opp").mean().alias("mean_pts_against"),
            pl.col("Score_opp").std().alias("std_pts_against"),
            pl.col("ScoreDiff").mean().alias("mean_score_diff"),
            pl.col("ScoreDiff").std().alias("std_score_diff"),

            # one-score + OT games
            ((pl.col("ScoreDiff").abs() <= 3) | (pl.col("NumOT") > 0)).mean().alias("close_games_pct"),
            pl.col("Win").filter((pl.col("ScoreDiff").abs() <= 3) | (pl.col("NumOT") > 0)).mean().alias("close_games_win_pct"),

            # shooting attempts
            (pl.col("FGA").mean() - pl.col("FGA3").mean()).alias("mean_FG2A"),
            pl.col("FGA3").mean().alias("mean_FG3A"),
            pl.col("FTA").mean().alias("mean_FTA"),

            # shooting percentages
            ((pl.col("FGM").sum() - pl.col("FGM3").sum()) / (pl.col("FGA").sum() - pl.col("FGA3").sum())).alias("FG2_pct"),
            (pl.col("FGM3").sum() / pl.col("FGA3").sum()).alias("FG3_pct"),
            (pl.col("FTM").sum() / pl.col("FTA").sum()).alias("FT_pct"),

            # total rebounds (offensive + defensive) per game
            (pl.col("OR") + pl.col("DR")).mean().alias("mean_reb"),
            (pl.col("OR") + pl.col("DR")).std().alias("std_reb"),
            (pl.col("OR_opp") + pl.col("DR_opp")).mean().alias("mean_reb_against"),
            (pl.col("OR_opp") + pl.col("DR_opp")).std().alias("std_reb_against"),

            # assists
            pl.col("Ast").mean().alias("mean_ast"),
            pl.col("Ast").std().alias("std_ast"),
            pl.col("Ast_opp").mean().alias("mean_ast_against"),
            pl.col("Ast_opp").std().alias("std_ast_against"),

            # turnovers
            pl.col("TO").mean().alias("mean_TO"),
            pl.col("TO").std().alias("std_TO"),
            pl.col("TO_opp").mean().alias("mean_TO_against"),
            pl.col("TO_opp").std().alias("std_TO_against"),

            # steals
            pl.col("Stl").mean().alias("mean_stl"),
            pl.col("Stl").std().alias("std_stl"),
            pl.col("Stl_opp").mean().alias("mean_stl_against"),
            pl.col("Stl_opp").std().alias("std_stl_against"),

            # blocks
            pl.col("Blk").mean().alias("mean_blk"),
            pl.col("Blk").std().alias("std_blk"),
            pl.col("Blk_opp").mean().alias("mean_blk_against"),
            pl.col("Blk_opp").std().alias("std_blk_against"),

            # fouls
            pl.col("PF").mean().alias("mean_fouls"),
            pl.col("PF").std().alias("std_fouls"),
            pl.col("PF_opp").mean().alias("mean_fouls_against"),
            pl.col("PF_opp").std().alias("std_fouls_against")]))

    # convert back to pandas
    return team_stats_pl.to_pandas()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def bayes_opt_nn(df, init_points=10, n_iter=100):
    """
    Perform Bayesian optimization to tune hyperparameters of a two-tower neural network for a binary classification task.

    Args:
    - df: DataFrame containing the features and labels
    - scaler: Scaler object to scale the features
    - init_points: Number of random points to sample before starting Bayesian optimization
    - n_iter: Number of optimization iterations

    Returns: 
    - None
    """

    # check for gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # assume 84 features: 42 for team X and 42 for team Y
    total_features = 84
    team_features = total_features // 2

    # sort the columns of X based on the last two characters of each column name (ensures that the team _X columns come before the team_Y columns)
    sorted_columns = sorted(df.columns, key=lambda col: col[-2:])
    df = df[sorted_columns]

    # define X and y
    X_df = df.drop(columns=['score_diff_adj_x', 'win_x'])
    y_numpy = df['win_x'].to_numpy()

    # split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_numpy, test_size=0.1, random_state=SEED)

    # scale
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # move to gpu   
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    # define the two-tower neural network
    class TwoTowerMLP(nn.Module):
        def __init__(self, team_features, hidden_size_x, hidden_size_y, combined_hidden, dropout_rate):
            super(TwoTowerMLP, self).__init__()
            # Team X branch
            self.team_x_branch = nn.Sequential(
                nn.Linear(team_features, int(hidden_size_x)),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(int(hidden_size_x), int(hidden_size_x) // 2),
                nn.ReLU())
            
            # Team Y branch
            self.team_y_branch = nn.Sequential(
                nn.Linear(team_features, int(hidden_size_y)),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(int(hidden_size_y), int(hidden_size_y) // 2),
                nn.ReLU())
            
            # combined layers: note the input is the concatenation of the two branch outputs
            combined_input_size = (int(hidden_size_x) // 2) + (int(hidden_size_y) // 2)
            self.combined_layers = nn.Sequential(
                nn.Linear(combined_input_size, int(combined_hidden)),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(int(combined_hidden), 1),
                nn.Sigmoid())
            
        def forward(self, x):
            # split input features into team X and team Y portions
            team_x_input = x[:, :team_features] # first 42 features
            team_y_input = x[:, team_features:] # last 42 features
            out_x = self.team_x_branch(team_x_input)
            out_y = self.team_y_branch(team_y_input)
            combined = torch.cat([out_x, out_y], dim=1)
            output = self.combined_layers(combined)
            return output

    # objective function for Bayesian optimization
    def objective(hidden_size_x, hidden_size_y, combined_hidden, dropout_rate, lr, weight_decay):
        # Convert parameters to appropriate types
        hidden_size_x = int(hidden_size_x)
        hidden_size_y = int(hidden_size_y)
        combined_hidden = int(combined_hidden)
        dropout_rate = float(dropout_rate)
        lr = float(lr)
        weight_decay = float(weight_decay)
        
        # instantiate the model with the given hyperparameters
        model = TwoTowerMLP(team_features, hidden_size_x, hidden_size_y, combined_hidden, dropout_rate).to(device)
        criterion = nn.BCELoss()  # binary cross entropy loss
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # train for a fixed number of epochs (adjust as needed)
        n_epochs = 1000
        model.train()
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            outputs = model(X_train_tensor).squeeze()  # shape: (batch_size,)
            loss = criterion(outputs, y_train_tensor.float())
            loss.backward()
            optimizer.step()
        
        # evaluate on the validation set
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor).squeeze()
            val_outputs_np = val_outputs.cpu().numpy()
            y_val_np = y_test_tensor.cpu().numpy()

            # compute log loss (lower is better)
            val_loss = log_loss(y_val_np, val_outputs_np)
        
        # return negative log loss because BayesianOptimization maximizes the objective
        return -val_loss

    # define the hyperparameter search space
    pbounds = {
        'hidden_size_x': (2, 256),     # first tower hidden size for team X
        'hidden_size_y': (2, 256),     # first tower hidden size for team Y
        'combined_hidden': (2, 256),   # hidden layer size after merging towers
        'dropout_rate': (0.1, 0.9),    # dropout rate
        'lr': (1e-5, 1e-1),            # learning rate
        'weight_decay': (1e-5, 1e-1)   # L2 regularization
    }

    # initialize the Bayesian optimizer
    optimizer_bo = BayesianOptimization(f=objective, pbounds=pbounds, random_state=SEED, verbose=2)

    # run the optimizer
    optimizer_bo.maximize(init_points=init_points, n_iter=n_iter)

    # extract best params
    best_params = optimizer_bo.max['params']

    # create and train best model
    final_model = TwoTowerMLP(
        team_features=team_features,
        hidden_size_x=int(best_params['hidden_size_x']),
        hidden_size_y=int(best_params['hidden_size_y']),
        combined_hidden=int(best_params['combined_hidden']),
        dropout_rate=float(best_params['dropout_rate'])).to(device)

    # define final loss and optimizer
    criterion = nn.BCELoss()
    adam = optim.Adam(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])

    # train final model
    final_model.train()
    for epoch in range(1000):  # more epochs for final model
        adam.zero_grad()
        outputs = final_model(X_train_tensor)  # outputs: shape (N, 1)
        
        # make sure targets have shape (N, 1)
        loss = criterion(outputs, y_train_tensor.float().unsqueeze(1))
        loss.backward()
        adam.step()

    # evaluate final model
    final_model.eval()
    with torch.no_grad():
        y_pred_prob = final_model(X_test_tensor).cpu().numpy()  # probabilities from sigmoid
    final_loss = log_loss(y_test, y_pred_prob)

    # for binary prediction, threshold at 0.5:
    final_predictions = (y_pred_prob >= 0.5).astype(int)
    final_acc = accuracy_score(y_test, final_predictions)
    print('Final log loss of best model:', final_loss)
    print('Final accuracy of best model:', final_acc)