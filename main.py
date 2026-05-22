from src.models.neural_net import NeuralNet
from src.utils import h
import pandas as pd
import joblib
import torch

"""
Parsing 2026 playoff data and make it readable for the model
Get the winner of each playoff matchup including the NBA Finals 
"""


STATS = [
    "PointsPG",
    "OppPointsPG",
    "DiffPointsPG",
    "E_OFF_RATING",
    "OFF_RATING",
    "E_DEF_RATING",
    "DEF_RATING",
    "E_NET_RATING",
    "NET_RATING",
    "AST_PCT",
    "AST_TO",
    "AST_RATIO",
    "OREB_PCT",
    "DREB_PCT",
    "REB_PCT",
    "TM_TOV_PCT",
    "EFG_PCT",
    "TS_PCT",
    "E_PACE",
    "PACE",
    "PIE",
]

playoff = pd.read_csv(
    "../NBA_Series_Predictor/src/data/processed/playoff_teams/playoff_teams_25_26.csv"
)

reg = pd.read_csv(
    "../NBA_Series_Predictor/src/data/processed/regular_season_matchups/regular_season_matchups_25_26.csv"
)


# create the first round matchups using data of the 16 playoff teams
def create_rows(conf, i, i2):

    test = []

    df = playoff[
        ((playoff["PlayoffRank"] == i) | (playoff["PlayoffRank"] == i2))
        & (playoff["Conference"] == conf)
    ]

    match = {
        "TEAM_HIGH_NAME": df.loc[df["PlayoffRank"] == i, "TEAM_NAME"].values[0],
        "TEAM_LOW_NAME": df.loc[df["PlayoffRank"] == i2, "TEAM_NAME"].values[0],
    }

    for col in STATS:
        high_val = df.loc[df["PlayoffRank"] == i, col].values[0]
        low_val = df.loc[df["PlayoffRank"] == i2, col].values[0]
        match[f"{col}_DIFF"] = high_val - low_val

    # calculate h2h win ratio
    high = df.loc[df["PlayoffRank"] == i, "TEAM_NAME"].values[0]
    low = df.loc[df["PlayoffRank"] == i2, "TEAM_NAME"].values[0]

    matchups = reg[
        ((reg["TEAM_NAME_HOME"] == high) & (reg["TEAM_NAME_AWAY"] == low))
        | ((reg["TEAM_NAME_HOME"] == low) & (reg["TEAM_NAME_AWAY"] == high))
    ]

    a_wins = len(
        matchups[
            ((matchups["TEAM_NAME_HOME"] == high) & (matchups["WL_HOME"] == "W"))
            | ((matchups["TEAM_NAME_AWAY"] == high) & (matchups["WL_AWAY"] == "W"))
        ]
    )

    match["H2H_WIN_RATIO"] = (a_wins / len(matchups)) if (a_wins / len(matchups)) > 0 else 0

    test.append(match)

    # preserve extra stats
    match["CONF"] = conf
    match["HIGH_SEED"] = i
    match["LOW_SEED"] = i2

    return pd.DataFrame(test)


# simulate the match between two teams
def sim(model, scaler, data, STATS, finals):
    data[STATS] = scaler.transform(data[STATS])
    X_test = h.convert(data, STATS)

    with torch.no_grad():
        output = model(X_test)
        prob = torch.sigmoid(output)
        pred = (output >= 0.5).float()
        data["LABEL"] = pred.cpu().numpy()
        data["PROB"] = prob.cpu().numpy()

    data["WINNER_NAME"] = data.apply(
        lambda x: x["TEAM_HIGH_NAME"] if x["LABEL"] == 1 else x["TEAM_LOW_NAME"], axis=1
    )
    data["LOSER_NAME"] = data.apply(
        lambda x: x["TEAM_HIGH_NAME"] if x["LABEL"] == 0 else x["TEAM_LOW_NAME"], axis=1
    )

    data["WINNER_SEED"] = data.apply(
        lambda x: x["HIGH_SEED"] if x["LABEL"] == 1 else x["LOW_SEED"], axis=1
    )
    data["LOSER_SEED"] = data.apply(
        lambda x: x["HIGH_SEED"] if x["LABEL"] == 0 else x["LOW_SEED"], axis=1
    )

    if finals:
        data["WINNER_CONF"] = data.apply(
            lambda x: x["HIGH_SEED_CONF"] if x["LABEL"] == 1 else x["LOW_SEED_CONF"],
            axis=1,
        )
        data["LOSER_CONF"] = data.apply(
            lambda x: x["HIGH_SEED_CONF"] if x["LABEL"] == 0 else x["LOW_SEED_CONF"],
            axis=1,
        )

        for i, x in data.iterrows():
            print(
                f"The {x['WINNER_CONF']}ern Conference Champion {x['WINNER_NAME']} defeated the {x['LOSER_CONF']}ern Conference Champion {x['LOSER_NAME']}"
            )
    else:
        for i, x in data.iterrows():
            print(
                f"The {x['WINNER_SEED']} seeded {x['WINNER_NAME']} defeated the {x['LOSER_SEED']} seeded {x['LOSER_NAME']}"
            )

    return data


# reduce to only the teams that won the current round
def reduce(teams, round):

    # dataframe of all teams that won the round
    winners = pd.DataFrame()
    winners["WINNER"] = teams.apply(
        lambda x: x["TEAM_HIGH_NAME"] if x["LABEL"] == 1 else x["TEAM_LOW_NAME"], axis=1
    )
    winners["WINNER_SEED"] = teams.apply(
        lambda x: x["HIGH_SEED"] if x["LABEL"] == 1 else x["LOW_SEED"], axis=1
    )
    winners["CONF"] = teams["CONF"]

    if round == "Finals":
        west = winners[winners["CONF"] == "West"].iloc[0]["WINNER"]
        east = winners[winners["CONF"] == "East"].iloc[0]["WINNER"]
        data = create_finals(west, east)
    else:
        west = winners[winners["CONF"] == "West"]
        east = winners[winners["CONF"] == "East"]
        data = make_matchups(west, east, round)

    return data


# create accurate matchups for the next round
def make_matchups(west, east, round):
    pairs = {}
    testing_data = pd.DataFrame()

    if round == "Conf-Semis":
        pairs = {(1, 4), (1, 5), (2, 3), (2, 6), (3, 7), (4, 8), (5, 8)}
    elif round == "Conf-Finals":
        pairs = {
            (1, 2),
            (1, 3),
            (1, 7),
            (1, 6),
            (2, 8),
            (3, 8),
            (6, 8),
            (7, 8),
            (2, 4),
            (3, 4),
            (4, 6),
            (4, 7),
            (2, 5),
            (3, 5),
            (5, 6),
            (5, 7),
        }

    # makes sure both teams are correctly matched up for the next round based on seed and conference
    for pair in pairs:
        top, bot = pair
        if (top in west["WINNER_SEED"].values) & (bot in west["WINNER_SEED"].values):
            testing_data = pd.concat(
                [testing_data, create_rows("West", top, bot)], ignore_index=True
            )

        if (top in east["WINNER_SEED"].values) & (bot in east["WINNER_SEED"].values):
            testing_data = pd.concat(
                [testing_data, create_rows("East", top, bot)], ignore_index=True
            )

    return testing_data


# create the nba finals matchup
def create_finals(east_winner, west_winner):

    east_wins = playoff.loc[playoff["TEAM_NAME"] == east_winner, "WINS"].values[0]
    west_wins = playoff.loc[playoff["TEAM_NAME"] == west_winner, "WINS"].values[0]

    # higher wins = high seed
    if west_wins >= east_wins:
        high, low = west_winner, east_winner
    else:
        high, low = east_winner, west_winner

    high_stats = playoff[playoff["TEAM_NAME"] == high].iloc[0]
    low_stats = playoff[playoff["TEAM_NAME"] == low].iloc[0]

    match = {"TEAM_HIGH_NAME": high, "TEAM_LOW_NAME": low}

    for col in STATS:
        match[f"{col}_DIFF"] = high_stats[col] - low_stats[col]

    matchups = reg[
        ((reg["TEAM_NAME_HOME"] == high) & (reg["TEAM_NAME_AWAY"] == low))
        | ((reg["TEAM_NAME_HOME"] == low) & (reg["TEAM_NAME_AWAY"] == high))
    ]
    a_wins = len(
        matchups[
            ((matchups["TEAM_NAME_HOME"] == high) & (matchups["WL_HOME"] == "W"))
            | ((matchups["TEAM_NAME_AWAY"] == high) & (matchups["WL_AWAY"] == "W"))
        ]
    )

    match["H2H_WIN_RATIO"] = (a_wins / len(matchups)) if (a_wins / len(matchups)) > 0 else 0

    # preserve extra stats
    match["HIGH_SEED_CONF"] = high_stats["Conference"]
    match["LOW_SEED_CONF"] = low_stats["Conference"]
    match["HIGH_SEED"] = high
    match["LOW_SEED"] = low

    return pd.DataFrame([match])


def main():
    STATS = [
        "PointsPG_DIFF",
        "OppPointsPG_DIFF",
        "DiffPointsPG_DIFF",
        "E_OFF_RATING_DIFF",
        "OFF_RATING_DIFF",
        "E_DEF_RATING_DIFF",
        "DEF_RATING_DIFF",
        "E_NET_RATING_DIFF",
        "NET_RATING_DIFF",
        "AST_PCT_DIFF",
        "AST_TO_DIFF",
        "AST_RATIO_DIFF",
        "OREB_PCT_DIFF",
        "DREB_PCT_DIFF",
        "REB_PCT_DIFF",
        "TM_TOV_PCT_DIFF",
        "EFG_PCT_DIFF",
        "TS_PCT_DIFF",
        "E_PACE_DIFF",
        "PACE_DIFF",
        "PIE_DIFF",
        "H2H_WIN_RATIO",
    ]

    # load model data based on validation accuracy
    model = NeuralNet()
    model.load_state_dict(
        torch.load(
            "/Users/Sean/PycharmProjects/NBA_Series_Predictor/src/data/model/best_model.pt"
        )
    )

    # place model in evaluation mode
    model.eval()

    # load scaler to transform testing data
    scaler = joblib.load(
        "/Users/Sean/PycharmProjects/NBA_Series_Predictor/src/data/model/scaler.pkl"
    )

    # first round
    teams = pd.DataFrame()
    for i in range(1, 5):
        teams = pd.concat(
            [teams, create_rows("West", i, 9 - i), create_rows("East", i, 9 - i)],
            ignore_index=True,
        )
    print("---FIRST ROUND---")
    teams = sim(model, scaler, teams, STATS, False)
    # print(teams.to_string())

    # conference semifinals
    teams = reduce(teams, "Conf-Semis")
    print("---Conference Semifinals---")
    teams = sim(model, scaler, teams, STATS, False)
    # print(teams.to_string())

    # conference finals
    teams = reduce(teams, "Conf-Finals")
    print("---Conference Finals---")
    teams = sim(model, scaler, teams, STATS, False)
    # print(teams.to_string())

    # nba finals
    teams = reduce(teams, "Finals")
    print("---NBA Finals---")
    teams = sim(model, scaler, teams, STATS, True)
    # print(teams.to_string())


if __name__ == "__main__":
    main()
