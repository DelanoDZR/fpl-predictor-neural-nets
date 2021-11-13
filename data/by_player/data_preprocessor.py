import pandas as pd


positions = {'GK': 'gks', 'DEF': 'defs', 'MID': 'mids', 'FWD': 'fwds'}


def split_into_positions():
    dataset = pd.read_csv(
            "data\\merged_gw.csv", sep=",", index_col=0)

    for position in positions:
        positionalSet = dataset[dataset.position == position]
        positionalSet.to_csv('data\\%s.csv' % positions[position])


def split_into_players():
    for position in positions:
        dataset = pd.read_csv(("data\\%s.csv" % positions[position]), sep=",")

    players = pd.unique(dataset['name'])
    for player in players:
        playerDataset = dataset[dataset['name'] == player]
        playerDataset.to_csv('data\\%s\\%s.csv' % (positions[position], player))


def enrich_with_target_output():
    for position in positions:
        dataset = pd.read_csv(("data\\by_player\\raw\\%s.csv" % positions[position]), sep=",", index_col='GW')

        players = pd.unique(dataset['name'])
        for player in players:
            player_dataset = dataset[dataset['name'] == player]
            player_dataset = player_dataset.drop_duplicates(['kickoff_time'])
            if player_dataset.shape[0] != 38:
                continue
            target_points = []

            gameweek = 0
            for index, row in player_dataset.iterrows():
                gameweek += 1
                if index < 38:
                    target_points.append(player_dataset.iloc[gameweek]['total_points'])

            target_player_dataset = player_dataset.copy().iloc[:-1, :]
            target_player_dataset['Target_Output'] = target_points
            target_player_dataset.to_csv('data\\by_player\\%s\\%s.csv' % (positions[position], player))
