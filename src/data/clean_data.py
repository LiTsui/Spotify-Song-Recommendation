from src.utils import h
import pandas as pd
import time

"""
Clean data for every NBA season from 1996 to 2026 to only show playoff team standings and playoff team regular season matchups
"""


# creates the playoffs file where there is the standings of each playoff team
def clean_standings(standings, team_stats, season):
    # merge rows if they have common TeamID/TEAM_ID
    playoff = standings.merge(
        team_stats, left_on="TeamID", right_on="TEAM_ID", how="inner"
    )

    # note that nba started using play-in tournament in 2019-20 so some PlayoffRanks need to be changed
    if season == "21_22":
        playoff.loc[14, "PlayoffRank"] = 9  # change Cleveland Cavaliers to 9th seed
        playoff.loc[15, "PlayoffRank"] = 9  # change Los Angeles Clippers to 9th seed
        playoff.loc[16, "PlayoffRank"] = 8  # change Atlanta Hawks to 8th seed
        playoff.loc[17, "PlayoffRank"] = 8  # change New Orleans Pelicans to 8th seed
    if season == "22_23":
        playoff.loc[12, "PlayoffRank"] = 8  # change Miami Heat to 8th seed
        playoff.loc[14, "PlayoffRank"] = 7  # change Atlanta Hawks to 7th seed
    if season == "23_24":
        playoff.loc[12, "PlayoffRank"] = 8  # change New Orleans Pelicans to 8th seed
        playoff.loc[14, "PlayoffRank"] = 7  # change Los Angeles Lakers to 7th seed
    if season == "24_25":
        playoff.loc[19, "PlayoffRank"] = 8  # change Miami Heat to 8th seed
        playoff.loc[15, "PlayoffRank"] = 10  # change Atlanta Hawks to 10th seed
    if season == "25_26":
        playoff.loc[14, "PlayoffRank"] = 7  # change Portland Trail Blazers to 7th seed
        playoff.loc[12, "PlayoffRank"] = 8  # change Phoenix Suns to 8th seed

    # old data from 1996 to 2001 has incorrect PlayoffRank
    if season == "96_97":
        playoff.loc[2, "PlayoffRank"] = 3  # change Houston Rockets to 3rd seed
        playoff.loc[10, "PlayoffRank"] = 6  # change Charlotte Hornets to 6th seed
        playoff.loc[11, "PlayoffRank"] = 7  # change Phoenix Suns to 7th seed
    if season == "97_98":
        print(playoff.to_string())
        playoff.loc[2, "PlayoffRank"] = 3  # change Los Angeles Lakers to 3rd seed
        playoff.loc[7, "PlayoffRank"] = 5  # change San Antonio Spurs to 5th seed
        playoff.loc[13, "PlayoffRank"] = 8  # change New Jersey Nets to 8th seed
    if season == "98_99":
        playoff.loc[1, "PlayoffRank"] = 3  # change Orlando Magic to 3rd seed
        playoff.loc[2, "PlayoffRank"] = 2  # change Indiana Pacers to 2nd seed
        playoff.loc[4, "PlayoffRank"] = 3  # change Utah Jazz to 3rd seed
        playoff.loc[5, "PlayoffRank"] = 2  # change Portland Trail Blazers to 2nd seed
        playoff.loc[8, "PlayoffRank"] = 5  # change Houston Rockets to 5th seed
        playoff.loc[11, "PlayoffRank"] = 7  # change Milwaukee Bucks to 7th seed
        playoff.loc[13, "PlayoffRank"] = 7  # change Phoenix Suns to 7th seed
        playoff.loc[16, "PlayoffRank"] = 9  # change Seattle Supersonics to 9th seed
    if season == "99_00":
        playoff.loc[6, "PlayoffRank"] = 5  # change Philadelphia 76ers to 5th seed
        playoff.loc[7, "PlayoffRank"] = 5  # change Phoenix Suns to 5th seed
        playoff.loc[12, "PlayoffRank"] = 8  # change Milwaukee Bucks to 8th seed
    if season == "00_01":
        playoff.loc[6, "PlayoffRank"] = 5  # change Dallas Mavericks to 5th seed

    # keep only the 16 playoff teams
    seed = [1, 2, 3, 4, 5, 6, 7, 8]
    playoff = playoff[playoff["PlayoffRank"].isin(seed)]

    # remove redundant or insignificant stats
    playoff = playoff.drop(
        columns=[
            "TEAM_ID",
            "GP",
            "W",
            "L",
            "W_PCT",
            "EliminatedConference",
            "EliminatedDivision",
            "ClinchedConferenceTitle",
            "ClinchedDivisionTitle",
            "ClinchIndicator",
            "LeagueRank",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "PostAS",
            "GP_RANK",
        ]
    )

    return playoff


def main():
    for i in range(1996, 2026):
        season = f"{str(i)[-2:]}_{str(i + 1)[-2:]}"  # follow format 00_01, 09_10
        print(f"Cleaning Data for {i}-{str(i + 1)[-2:]}")

        standings = pd.read_csv(f"../data/raw/standings/standings_{season}.csv")
        team_stats = pd.read_csv(f"../data/raw/team_stats/team_stats_{season}.csv")
        game_logs = pd.read_csv(f"../data/raw/game_logs/game_logs_{season}.csv")

        # keeps all the relevant data necessary to draw conclusions team matchups
        playoff = clean_standings(standings, team_stats, season)

        # keeps all regular season matchups that involve playoff teams
        matchups = h.clean_matchups(game_logs, playoff)

        playoff.to_csv(
            f"../data/processed/playoff_teams/playoff_teams_{season}.csv",
            index=False,
        )
        matchups.to_csv(
            f"../data/processed/regular_season_matchups/regular_season_matchups_{season}.csv",
            index=False,
        )

        time.sleep(5)  # necessary to avoid time out


if __name__ == "__main__":
    main()
