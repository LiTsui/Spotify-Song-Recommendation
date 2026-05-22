from nba_api.stats.endpoints import leaguestandings, leaguedashteamstats, leaguegamefinder
import time

"""
Loads data for every NBA season from 1996 to 2026
"""


# load standings, team stats, game logs from each season in the NBA
def load(season):

    # get league standings
    standings = leaguestandings.LeagueStandings(season=season).get_data_frames()[0]
    standings.to_csv(
        f"../data/raw/standings/standings_{season[2:4]}_{season[-2:]}.csv",
        index=False,
    )

    # get team stats
    team_stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Advanced",
    ).get_data_frames()[0]
    team_stats.to_csv(
        f"../data/raw/team_stats/team_stats_{season[2:4]}_{season[-2:]}.csv",
        index=False,
    )

    # get game logs
    games = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable="Regular Season",
        player_or_team_abbreviation="T",  # team
    ).get_data_frames()[0]
    games.to_csv(
        f"../data/raw/game_logs/game_logs_{season[2:4]}_{season[-2:]}.csv",
        index=False,
    )

    # get playoff history
    playoff_df = leaguegamefinder.LeagueGameFinder(
        season_nullable=season, league_id_nullable="00", season_type_nullable="Playoffs"
    ).get_data_frames()[0]
    playoff_df.to_csv(
        f"../data/raw/playoff_history/playoff_history_{season[2:4]}_{season[-2:]}.csv",
        index=False,
    )


# load all data for all seasons from 1996 to 2025 where 1996 is the start of the play by play era
def main():
    for i in range(1996, 2026):
        season = f"{i}-{str(i + 1)[-2:]}"  # follow format 2000-01, 2009-10
        print(f"Loading {season}")
        load(season)
        print("Raw data saved")
        time.sleep(5)  # necessary to avoid time out


if __name__ == "__main__":
    main()
