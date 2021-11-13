import data
import learn
import models
import numpy as np
import pandas as pd


def main():
    positions = {'GK': 'gks', 'DEF': 'defs', 'MID': 'mids', 'FWD': 'fwds'}

    #Plug into shape
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
            try:
                target_player_dataset['Target_Output'] = target_points
            except:
                print("oops" + player)
            target_player_dataset.to_csv('data\\by_player\\%s\\%s.csv' % (positions[position], player))



    """   
    Target Shape (NONE, NUMBER_OF_GAMEWEEKS, NUMBER_OF_ATTRIBUTES)
    i.e (NONE, 5, 23)
    [
    [sonGW1[form, score], sonGW2[form, score]],
    [kaneGW1[form, score], kaneGW2[form, score]]
    ]
    """


    # train_labels = train.pop('Target_Output')
    # train_features = np.array([train.to_numpy()], dtype=np.float)
    # train_labels = np.array([train_labels], dtype=np.float)
    """
    train_features, train_labels, test_features, test_labels = data.get_data_sets()

    model = models.mlp((1, 23))
    learn.train_and_evaluate(model, train_features, train_labels, test_features, test_labels, 100)
    
    print("Prediction")
    print(test_features)
    print(test_features.shape)
    #print(model.predict(np.array(tester)))
    """

if __name__ == "__main__":
    main()
