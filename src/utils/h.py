import torch
import pandas as pd

"""
Stores all functions used across multiple files
"""


# creates the regular_season_matchups file which keeps only regular_season_matchups between teams that are in the playoffs
def clean_matchups(game_logs, playoff):
    # split dataframe into home and away teams
    away_df = game_logs[game_logs["MATCHUP"].str.contains("@")].copy()
    home_df = game_logs[game_logs["MATCHUP"].str.contains("vs.")].copy()

    # extract abbreviations of the opponent for each row
    away_df["OPP_ABBR"] = away_df["MATCHUP"].str.split().str[-1]
    home_df["OPP_ABBR"] = home_df["MATCHUP"].str.split().str[-1]

    # combine the home and away dataframes
    matchups = home_df.merge(
        away_df,
        left_on=["GAME_DATE", "OPP_ABBR", "TEAM_ABBREVIATION"],
        right_on=["GAME_DATE", "TEAM_ABBREVIATION", "OPP_ABBR"],
        suffixes=("_HOME", "_AWAY"),
    )

    # remove teams that are not in the 2025-26 NBA playoffs
    matchups = matchups[
        matchups["TEAM_NAME_HOME"].isin(playoff["TEAM_NAME"])
        & matchups["TEAM_NAME_AWAY"].isin(playoff["TEAM_NAME"])
    ]

    return matchups


# converts dataframe values into tensor
def convert(df, label):
    return torch.tensor(df[label].to_numpy(), dtype=torch.float32)


# helper function to compare predicted vs actual and the probability
def actual_vs_pred(model, X_valid, y_valid, pred_val):
    with torch.no_grad():
        probs = torch.sigmoid(model(X_valid).squeeze())

    debug_df = pd.DataFrame()
    debug_df["LABEL"] = y_valid.squeeze().numpy()
    debug_df["PROB"] = probs.numpy()
    debug_df["PRED"] = pred_val.numpy()

    print(debug_df[["LABEL", "PRED", "PROB"]])
