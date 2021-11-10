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

