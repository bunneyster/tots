import contextlib
import csv
import json
import os
import random

import constants

TRAIN_DATA_PATH = './matches/csv/train_data.csv'
TEST_DATA_PATH = './matches/csv/test_data.csv'

def partition_players(all_players):
    won_players = []
    lost_players = []
    for player in all_players:
        if player['winningTeam']:
            won_players.append(player)
        else:
            lost_players.append(player)
    return won_players, lost_players

class MatchValidator(object):
    """Validate data types of the given match JSON."""
    def __init__(self, match):
        super(MatchValidator, self).__init__()
        self.match = match

    def map_name(self):
        return self.match['metadata']['mapName']

    def duration(self):
        return self.match['metadata']['durationSeconds']

    def players(self):
        return self.match['players'] or []

    def map_is_valid(self):
        return self.map_name() in constants.MAPS

    def duration_is_valid(self):
        return isinstance(self.duration(), int)

    def mmr_is_valid(self, player):
        return isinstance(player['mmr']['starting'], int)

    def hero_is_valid(self, player):
        return player['hero'] in constants.HEROES

    def players_are_valid(self):
        for player in self.players():
            if not (self.mmr_is_valid(player) and self.hero_is_valid(player)):
                return False
        return len(self.players()) == 10

    def is_valid(self):
        return self.map_is_valid() and self.duration_is_valid() and self.players_are_valid()


saved_replay_ids = set()
def write_matches_to_csv(file_name, matches):
    with open(file_name, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(matches)):
            validator = MatchValidator(matches[i])
            replay_id = matches[i]['metadata']['replayId']
            if (not validator.is_valid()) or (replay_id in saved_replay_ids):
                continue

            won_players, lost_players = partition_players(matches[i]['players'])
            won_heroes = [player['hero'] for player in won_players]
            lost_heroes = [player['hero'] for player in lost_players]
            won_mmrs = [player['mmr']['starting'] for player in won_players]
            lost_mmrs = [player['mmr']['starting'] for player in lost_players]
            map_name = [matches[i]['metadata']['mapName']]
            duration = [matches[i]['metadata']['durationSeconds']]

            winners_vector = [
                won_heroes, lost_heroes, won_mmrs, lost_mmrs, map_name, duration, ['won']
            ]
            losers_vector = [
                lost_heroes, won_heroes, lost_mmrs, won_mmrs, map_name, duration, ['lost']
            ]

            flat_winners_vector = [value for feature_list in winners_vector for value in feature_list]
            flat_losers_vector = [value for feature_list in losers_vector for value in feature_list]

            writer.writerow(flat_winners_vector)
            writer.writerow(flat_losers_vector)
            saved_replay_ids.add(replay_id)


with contextlib.suppress(FileNotFoundError):
    os.remove(TRAIN_DATA_PATH)
    os.remove(TEST_DATA_PATH)

json_dir = './matches/json'
json_files = sorted(os.listdir(json_dir), key=lambda t: int(os.path.splitext(t)[0]))
for basename in json_files:
    with open(os.path.join(json_dir, basename)) as data_file:
        print(f"Reading file {basename}")
        matches = json.load(data_file)
        matches = [m for m in matches if MatchValidator(m).is_valid()]
        random.shuffle(matches)
        write_matches_to_csv(TRAIN_DATA_PATH, matches[:len(matches) // 10 * 9])
        write_matches_to_csv(TEST_DATA_PATH, matches[len(matches) // 10 * 9:])
        replay_count = len(saved_replay_ids)
        print(f"Converted {replay_count} replays.\n")
