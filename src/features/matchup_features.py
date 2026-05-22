from src.utils import h
import pandas as pd
import time

"""
Combine statistical data in every playoff series and indicates which team won which series from 1996 to 2025. 
Note: The 2025-2026 playoffs are what is to be predicted.
"""


# compares the stats between teams in previous playoff matchups with winner - loser to reflect labels
def make_matchup(x, playoff):
    stat = [
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

    # get the name for the winner and loser of each series
    winner = find_winner_or_loser_series(x, y="W")
    loser = find_winner_or_loser_series(x, y="L")

    # gets the statistical data of each playoff team for that season
    win = playoff.loc[playoff["TEAM_NAME"] == winner].iloc[0]
    lose = playoff.loc[playoff["TEAM_NAME"] == loser].iloc[0]

    # each column contains the difference between the higher seed and lower seed's stats for each playoff matchup
    d = {}
    for col in stat:
        d[f"{col}_DIFF"] = win[col] - lose[col]

    return pd.Series(d)


# finds which team is the higher seed or lower seed
def find_seed(x, y):
    if y == "higher":
        if x["ROUND"] == "NBA Finals":
            if x["WINNER_WINS"] >= x["LOSER_WINS"]:
                return x["SERIES_WINNER"]
            elif x["WINNER_WINS"] < x["LOSER_WINS"]:
                return x["SERIES_LOSER"]
        else:
            if x["WINNER_SEED"] < x["LOSER_SEED"]:
                return x["SERIES_WINNER"]
            else:
                return x["SERIES_LOSER"]
    elif y == "lower":
        if x["ROUND"] == "NBA Finals":
            if x["WINNER_WINS"] > x["LOSER_WINS"]:
                return x["SERIES_LOSER"]
            elif x["WINNER_WINS"] <= x["LOSER_WINS"]:
                return x["SERIES_WINNER"]
        else:
            if x["WINNER_SEED"] < x["LOSER_SEED"]:
                return x["SERIES_LOSER"]
            else:
                return x["SERIES_WINNER"]
    return None


# find winner or loser of a game depending on find
def find_winner_or_loser_game(x, y):
    if x["WL_HOME"] == y:
        return x["TEAM_NAME_HOME"]
    else:
        return x["TEAM_NAME_AWAY"]


# find winner or loser of a series depending on find
def find_winner_or_loser_series(x, y):
    if y == 'W':
        return x["SERIES_WINNER"]
    else:
        return x["SERIES_LOSER"]


# finds an element of each team in each matchup depending on find
def get_elem(x, team, playoff, find):
    match = playoff[playoff["TEAM_NAME"] == x[team]]
    if not match.empty:
        return match.iloc[0][find]
    return None


# finds the round that each matchup is held in
def get_round(x):
    w = int(x["WINNER_SEED"])
    l = int(x["LOSER_SEED"])

    pair = tuple(sorted((w, l)))

    if pair in {
        (1, 8),
        (2, 7),
        (3, 6),
        (4, 5),
    }:  #  first round is guaranteed to have these matchups
        return "First Round"
    elif pair in {(1, 4), (1, 5), (2, 3), (2, 6), (3, 7), (4, 8), (5, 8), (6, 7)}:
        return "Conf Semis"
    elif x["WINNER_CONF"] != x["LOSER_CONF"]:
        return "NBA Finals"
    else:
        return "Conf Finals"


# finds the winner of the series
def find_series_winner(x, cleaned_matchups):
    team1 = x["TEAM_NAME_HOME"]
    team2 = x["TEAM_NAME_AWAY"]

    # get only the matchup between team1 and team2 specifically
    sub_matchup = cleaned_matchups[
        (
            (cleaned_matchups["TEAM_NAME_HOME"] == team1)
            & (cleaned_matchups["TEAM_NAME_AWAY"] == team2)
        )
        | (
            (cleaned_matchups["TEAM_NAME_HOME"] == team2)
            & (cleaned_matchups["TEAM_NAME_AWAY"] == team1)
        )
    ]

    # get the team1 wins that are both at home and away
    team1_wins = (
        ((sub_matchup["TEAM_NAME_HOME"] == team1) & (sub_matchup["WL_HOME"] == "W"))
        | ((sub_matchup["TEAM_NAME_AWAY"] == team1) & (sub_matchup["WL_AWAY"] == "W"))
    ).sum()

    # get the team2 wins that are both at home and away
    team2_wins = (
        ((sub_matchup["TEAM_NAME_HOME"] == team2) & (sub_matchup["WL_HOME"] == "W"))
        | ((sub_matchup["TEAM_NAME_AWAY"] == team2) & (sub_matchup["WL_AWAY"] == "W"))
    ).sum()

    if team1_wins > team2_wins:
        return team1
    else:
        return team2


# get regular season matchup records between two teams
def get_h2h_record(x, regular_season, find):

    win = x["SERIES_WINNER"]
    loss = x["SERIES_LOSER"]

    h2h = regular_season[
        (
            (regular_season["TEAM_NAME_HOME"] == win)
            & (regular_season["TEAM_NAME_AWAY"] == loss)
        )
        | (
            (regular_season["TEAM_NAME_HOME"] == loss)
            & (regular_season["TEAM_NAME_AWAY"] == win)
        )
    ]

    high_wins = (
        ((h2h["TEAM_NAME_HOME"] == win) & (h2h["WL_HOME"] == "W"))
        | ((h2h["TEAM_NAME_AWAY"] == win) & (h2h["WL_AWAY"] == "W"))
    ).sum()

    low_wins = (
        ((h2h["TEAM_NAME_HOME"] == loss) & (h2h["WL_HOME"] == "W"))
        | ((h2h["TEAM_NAME_AWAY"] == loss) & (h2h["WL_AWAY"] == "W"))
    ).sum()

    if find == "w":
        return high_wins
    else:
        return low_wins


def main():
    train_df = pd.DataFrame()
    valid_df = pd.DataFrame()
    test_df = pd.DataFrame()

    for i in range(1996, 2025):
        season = f"{i}-{str(i + 1)[-2:]}"  # follow format 1996-97
        read_season = (
            f"{str(i)[-2:]}_{str(i + 1)[-2:]}"  # follow 96_97 for file reading/writing
        )

        print(f"Creating Playoff Matchup Data for {season}")

        # read playoff matches history and playoff teams data
        games = pd.read_csv(
            f"../data/raw/playoff_history/playoff_history_{read_season}.csv"
        )
        playoff = pd.read_csv(
            f"../data/processed/playoff_teams/playoff_teams_{read_season}.csv"
        )

        cleaned_matchups = h.clean_matchups(games, playoff)

        # find winner and loser of each game and use that to find the winner of each series
        cleaned_matchups["WINNER"] = cleaned_matchups.apply(
            find_winner_or_loser_game, axis=1, y="W"
        )
        cleaned_matchups["LOSER"] = cleaned_matchups.apply(
            find_winner_or_loser_game, axis=1, y="L"
        )
        cleaned_matchups["SERIES_WINNER"] = cleaned_matchups.apply(
            find_series_winner, axis=1, cleaned_matchups=cleaned_matchups
        )
        cleaned_matchups["SERIES_LOSER"] = cleaned_matchups.apply(
            lambda x: (
                x["TEAM_NAME_AWAY"]
                if x["TEAM_NAME_HOME"] == x["SERIES_WINNER"]
                else x["TEAM_NAME_HOME"]
            ),
            axis=1,
        )

        # keep only the following columns
        re_clean = cleaned_matchups[
            [
                "TEAM_ABBREVIATION_HOME",
                "TEAM_NAME_HOME",
                "GAME_DATE",
                "TEAM_ABBREVIATION_AWAY",
                "TEAM_NAME_AWAY",
                "WINNER",
                "SERIES_WINNER",
                "LOSER",
                "SERIES_LOSER",
            ]
        ].copy()

        # series key tuple compromising of the two team's ids
        re_clean["SERIES_KEY"] = cleaned_matchups.apply(
            lambda r: tuple(sorted([r["TEAM_ID_HOME"], r["TEAM_ID_AWAY"]])), axis=1
        )

        # reduce to the following columns
        matchup_data = re_clean[["SERIES_WINNER", "SERIES_LOSER"]].copy()

        # checks true for each game the winner of the series has at home and false for each game the loser of the series has at home
        matchup_data["WINNER_HOME_GAMES"] = re_clean.apply(
            lambda r: r["SERIES_WINNER"] == r["TEAM_NAME_HOME"]
            and r["SERIES_LOSER"] == r["TEAM_NAME_AWAY"],
            axis=1,
        )

        # checks true for each game the winner of the series won and false for each game the winner of the series lost
        matchup_data["WINNER_GAMES_WON"] = re_clean.apply(
            lambda r: r["SERIES_WINNER"] == r["WINNER"]
            and (
                (
                    r["SERIES_WINNER"] == r["TEAM_NAME_HOME"]
                    and r["SERIES_LOSER"] == r["TEAM_NAME_AWAY"]
                )
                or (
                    r["SERIES_LOSER"] == r["TEAM_NAME_HOME"]
                    and r["SERIES_WINNER"] == r["TEAM_NAME_AWAY"]
                )
            ),
            axis=1,
        )

        # finds number of games both teams have won in that specific series
        x = (
            matchup_data.groupby(["SERIES_WINNER", "SERIES_LOSER"])["WINNER_GAMES_WON"]
            .agg(WINNER_GAMES_WON="sum", LOSER_GAMES_WON=lambda s: (~s).sum())
            .reset_index()
        )

        # finds number of home games both teams played in that specific series
        y = (
            matchup_data.groupby(["SERIES_WINNER", "SERIES_LOSER"])["WINNER_HOME_GAMES"]
            .agg(WINNER_HOME_GAMES="sum", LOSER_HOME_GAMES=lambda s: (~s).sum())
            .reset_index()
        )

        # merge x and y to get SERIES_WINNER, SERIES_LOSER, WINNER_GAMES_WON, LOSER_GAMES_WON
        matchup_data = x.merge(y)

        # uses playoff data to find what the seeds are
        matchup_data["WINNER_SEED"] = matchup_data.apply(
            get_elem, axis=1, team="SERIES_WINNER", playoff=playoff, find="PlayoffRank"
        )
        matchup_data["LOSER_SEED"] = matchup_data.apply(
            get_elem, axis=1, team="SERIES_LOSER", playoff=playoff, find="PlayoffRank"
        )

        # needed in order to conclude which team is the top_seed and low_seed
        matchup_data["WINNER_CONF"] = matchup_data.apply(
            get_elem, axis=1, team="SERIES_WINNER", playoff=playoff, find="Conference"
        )
        matchup_data["LOSER_CONF"] = matchup_data.apply(
            get_elem, axis=1, team="SERIES_LOSER", playoff=playoff, find="Conference"
        )
        matchup_data["WINNER_WINS"] = matchup_data.apply(
            get_elem, axis=1, team="SERIES_WINNER", playoff=playoff, find="WINS"
        )
        matchup_data["LOSER_WINS"] = matchup_data.apply(
            get_elem, axis=1, team="SERIES_LOSER", playoff=playoff, find="WINS"
        )

        # finds the round of each playoff series
        matchup_data["ROUND"] = matchup_data.apply(get_round, axis=1)

        # labels which team is the higher and lower seeds
        matchup_data["HIGHER_SEED"] = matchup_data.apply(find_seed, axis=1, y="higher")
        matchup_data["LOWER_SEED"] = matchup_data.apply(find_seed, axis=1, y="lower")

        # print(matchup_data.to_string(index=False))

        stats = matchup_data.apply(make_matchup, axis=1, playoff=playoff)
        matchup_data = matchup_data.join(stats)

        # print(matchup_data.to_string())

        regular_season = pd.read_csv(
            f"../data/processed/regular_season_matchups/regular_season_matchups_{read_season}.csv"
        )

        matchup_data["SERIES_WINNER_H2H_WINS"] = matchup_data.apply(
            get_h2h_record, axis=1, regular_season=regular_season, find="w"
        )
        matchup_data["SERIES_LOSER_H2H_WINS"] = matchup_data.apply(
            get_h2h_record, axis=1, regular_season=regular_season, find="l"
        )

        # calculate h2h win ratio
        matchup_data["H2H_WIN_RATIO"] = matchup_data["SERIES_WINNER_H2H_WINS"] / (
                matchup_data["SERIES_WINNER_H2H_WINS"] + matchup_data["SERIES_LOSER_H2H_WINS"] + 1e-6
        )

        # preserve which data points are from which seasons for analysis
        matchup_data["SEASON"] = season

        matchup_data.to_csv(
            f"../data/processed/playoff_history_cleaned/playoff_history_cleaned_{read_season}.csv",
            index=False,
        )

        if i in range(1996, 2019):
            print(f"Adding {season} to Training Data")
            train_df = pd.concat([train_df, matchup_data])

        if i in range(2019, 2022):
            print(f"Adding {season} to Validation Data")
            valid_df = pd.concat([valid_df, matchup_data])

        if i in range(2022, 2025):
            print(f"Adding {season} to Test Data")
            test_df = pd.concat([test_df, matchup_data])

        time.sleep(5)

    # print(training_df.to_string())
    # print(valid_df.to_string())
    # print(test_df.to_string())

    # assign labels (binary classifier) where 1 means the higher seed won and 0 means the lower seed won
    train_df["LABEL"] = train_df.apply(
        lambda x: 1 if x["HIGHER_SEED"] == x["SERIES_WINNER"] else 0, axis=1
    )
    valid_df["LABEL"] = valid_df.apply(
        lambda x: 1 if x["HIGHER_SEED"] == x["SERIES_WINNER"] else 0, axis=1
    )
    test_df["LABEL"] = test_df.apply(
        lambda x: 1 if x["HIGHER_SEED"] == x["SERIES_WINNER"] else 0, axis=1
    )

    train_df.to_csv(f"../data/training.csv", index=False)
    valid_df.to_csv(f"../data/validation.csv", index=False)
    test_df.to_csv(f"../data/testing.csv", index=False)


if __name__ == "__main__":
    main()
