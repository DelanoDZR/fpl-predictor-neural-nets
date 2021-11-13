import pandas as pd
import numpy as np
import os


def get_data_sets(position):
    all_features = []
    all_labels = []
    a_directory = "data\\by_player\\%s" % position

    for filename in os.listdir(a_directory):
        filepath = os.path.join(a_directory, filename)
        dataset = pd.read_csv(
            filepath, sep=",")
        dataset = dataset.drop(
            columns=['name', 'GW', 'was_home', 'transfers_in', 'transfers_out', 'transfers_balance', 'selected',
                     'round', 'opponent_team', 'kickoff_time', 'fixture', 'position', 'team'])

        player_gameweeks_labels = dataset.pop('Target_Output')

        player_gameweeks_features = dataset[:3] #Pull 3 gameweeks per player
        player_gameweeks_labels = player_gameweeks_labels[4] #Target the fourth gameweek

        all_features.append(player_gameweeks_features)
        all_labels.append(player_gameweeks_labels)

    #Split the data into training and test sets
    train_features = np.array(all_features[:20], dtype=np.float)
    train_labels = np.array(all_labels[:20], dtype=np.float)

    test_features = np.array(all_features[21:30], dtype=np.float)
    test_labels = np.array(all_labels[21:30], dtype=np.float)

    return train_features, train_labels, test_features, test_labels
