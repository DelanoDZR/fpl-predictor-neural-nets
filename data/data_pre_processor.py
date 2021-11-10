import pandas as pd


def create_empty_gw_csvs():
    for i in range(1,39):
        open('raw\\gw%s.csv' %(i), 'x')


positions = {'GK': 'gks', 'DEF': 'defs', 'MID': 'mids', 'FWD': 'fwds'}


def mangle_data(position, week):

    def generate_gw_csvs(week):
        return 'gw%s.csv' %(str(week))

    this_week_csv_name = generate_gw_csvs(week)
    next_week_csv_name = generate_gw_csvs(week + 1)

    path_to_this_week_raw_csv = 'data\\raw\\%s' % this_week_csv_name
    path_to_next_week_raw_csv = 'data\\raw\\%s' % next_week_csv_name

    this_week_data = pd.read_csv(
        path_to_this_week_raw_csv, sep=",")
    this_week_data = this_week_data[this_week_data.position == position]
    this_week_data = this_week_data.drop_duplicates('name')

    next_week_data = pd.read_csv(
        path_to_next_week_raw_csv, sep=",")
    next_week_data = next_week_data.drop(next_week_data.columns.difference(['total_points','name']),1)
    next_week_data = next_week_data.drop_duplicates('name')
    next_week_data = next_week_data.rename(columns={'total_points':"Target_Output"})

    result = pd.merge(this_week_data, next_week_data, on=['name'])
    result = result.drop(columns=['transfers_in', 'transfers_out', 'transfers_balance', 'selected', 'round', 'penalties_saved', 'opponent_team', 'kickoff_time', 'fixture', 'position', 'team'])

    result = result.replace({'was_home': {True: 1, False: 0}})

    result.to_csv('data\\%s\\%s' % (positions[position], this_week_csv_name))


def prep_data():
    for i in range(1, 38):
        week = i
        for position in positions:
            mangle_data(position, week)
